from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from ..core.auth import get_user_id, require_api_key
from ..core.config import get_settings
from ..domain.schemas import ObservationBatchCreate, ObservationCreate, QuestionnaireSubmission, ReportCandidate, ReportCandidateConfirm, UserProfile
from ..integrations.apple_health_export import parse_apple_health_export
from ..integrations.report_ocr import extract_report_candidates
from ..repositories.database import SQLiteDatabase
from ..services.observations import ObservationService

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
settings = get_settings()
db = SQLiteDatabase(settings.database_path)
observations = ObservationService(db)


def set_runtime(runtime_settings, runtime_db: SQLiteDatabase) -> None:
    global settings, db, observations
    settings = runtime_settings
    db = runtime_db
    observations = ObservationService(db)


@router.post("/ingest/batches")
def ingest_batch(payload: ObservationBatchCreate, user_id: str = Depends(get_user_id)) -> dict:
    return observations.add_batch(user_id, payload.provider, payload.batch_id, payload.observations)


@router.post("/ingest/apple-health-export")
async def ingest_apple_health_export(file: UploadFile = File(...),
                                     batch_id: str = Header(min_length=8, max_length=128, alias="X-Import-Batch-ID"),
                                     user_id: str = Depends(get_user_id)) -> dict:
    if file.content_type not in {"application/xml", "text/xml", "application/octet-stream"} and not (file.filename or "").casefold().endswith(".xml"):
        raise HTTPException(status_code=415, detail="Expected an Apple Health export XML file")
    content = await file.read(50 * 1024 * 1024 + 1)
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="XML export exceeds 50 MB request limit; split it into smaller batches")
    try:
        parsed, report = parse_apple_health_export(content, batch_id)
    except ET.ParseError as error:
        raise HTTPException(status_code=422, detail={"code":"APPLE_HEALTH_XML_INVALID","message":str(error)})
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code":"APPLE_HEALTH_XML_INVALID","message":str(error)})
    if not parsed:
        return {"status":"NO_SUPPORTED_RECORDS","batch_id":batch_id,"parser":report,
                "message":"No supported observation types were found; unsupported records were not converted."}
    response = observations.add_batch(user_id, "apple_health_export_xml", batch_id, parsed)
    return {**response,"parser":report,"source_format":"Apple Health Export XML",
            "limitations":["Imports a user-provided export; native HealthKit permissions and background delivery require an iOS app.",
                           "Only mapped types are imported. Unmapped records are counted and omitted.",
                           "Sleep samples are stored as intervals by stage; sleep totals and derived scores remain uncomputed."]}


@router.post("/ingest/manual")
def ingest_manual(payload: ObservationBatchCreate, user_id: str = Depends(get_user_id)) -> dict:
    if payload.provider.casefold() not in {"manual", "questionnaire"}:
        raise HTTPException(status_code=422,detail="Manual endpoint accepts provider=manual or questionnaire")
    return observations.add_batch(user_id,payload.provider,payload.batch_id,payload.observations)


@router.put("/profiles/me")
def upsert_profile(profile: UserProfile, user_id: str = Depends(get_user_id)) -> dict:
    payload = profile.model_dump(mode="json") | {"user_id":user_id}
    with db.connect() as conn:
        conn.execute("INSERT INTO profiles(user_id,profile_json) VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json,updated_at=CURRENT_TIMESTAMP",
                     (user_id,json.dumps(payload,ensure_ascii=False)))
    return {"status":"saved","profile":payload}


@router.get("/profiles/me")
def get_profile(user_id: str = Depends(get_user_id)) -> dict:
    with db.connect() as conn:
        row=conn.execute("SELECT profile_json FROM profiles WHERE user_id=?",(user_id,)).fetchone()
    return {"status":"found","profile":json.loads(row[0])} if row else {"status":"not_found","profile":None}


@router.post("/questionnaires/submissions")
def submit_questionnaire(payload: QuestionnaireSubmission, user_id: str = Depends(get_user_id)) -> dict:
    submission_id=hashlib.sha256(f"{user_id}|{payload.instrument}|{payload.version}|{payload.completed_at.isoformat()}".encode()).hexdigest()
    with db.connect() as conn:
        existing=conn.execute("SELECT submission_id,status FROM questionnaire_submissions WHERE submission_id=?",(submission_id,)).fetchone()
        if existing:
            return {"submission_id":submission_id,"status":"DUPLICATE","scored":False}
        conn.execute("INSERT INTO questionnaire_submissions(submission_id,user_id,instrument,version,completed_at,answers_json,status) VALUES(?,?,?,?,?,?,?)",
                     (submission_id,user_id,payload.instrument,payload.version,payload.completed_at.astimezone(timezone.utc).isoformat(),json.dumps(payload.answers,ensure_ascii=False),"RECEIVED_NOT_SCORED"))
    return {"submission_id":submission_id,"status":"RECEIVED_NOT_SCORED","scored":False,
            "message":"Self-report is stored separately and is not converted into a sensor measurement."}


@router.get("/questionnaires/submissions")
def list_questionnaires(user_id: str = Depends(get_user_id)) -> dict:
    with db.connect() as conn:
        rows=conn.execute("SELECT * FROM questionnaire_submissions WHERE user_id=? ORDER BY completed_at DESC",(user_id,)).fetchall()
    return {"items":[{"submission_id":r["submission_id"],"instrument":r["instrument"],"version":r["version"],
                       "completed_at":r["completed_at"],"answers":json.loads(r["answers_json"]),"status":r["status"]} for r in rows]}


def _report_user_storage_key(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def _valid_report_signature(content_type: str, content: bytes) -> bool:
    if content_type == "application/pdf":
        return content.startswith(b"%PDF-")
    if content_type == "image/png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if content_type == "image/jpeg":
        return content.startswith(b"\xff\xd8\xff")
    return False


@router.post("/reports/upload-intake")
async def report_upload_intake(file: UploadFile=File(...),user_id:str=Depends(get_user_id))->dict:
    allowed={"application/pdf","image/jpeg","image/png"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415,detail="Only PDF, JPEG or PNG reports are accepted")
    if not file.filename or Path(file.filename).suffix.lower() not in {".pdf", ".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=415, detail="Report filename must end with .pdf, .jpg, .jpeg or .png")
    expected_type = {".pdf":"application/pdf", ".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".png":"image/png"}[Path(file.filename).suffix.lower()]
    if file.content_type != expected_type:
        raise HTTPException(status_code=415, detail="Report file extension and content type do not match")
    content=await file.read(15*1024*1024+1)
    if len(content)>15*1024*1024:
        raise HTTPException(status_code=413,detail="Report exceeds the 15 MB upload limit")
    if not _valid_report_signature(expected_type,content):
        raise HTTPException(status_code=415,detail="Report content does not match the declared file type")
    digest=hashlib.sha256(content).hexdigest()
    job_id=str(hashlib.sha256(f"{user_id}|{digest}".encode()).hexdigest())
    reports_dir=settings.normalized_data_dir.parent/"report_uploads"/_report_user_storage_key(user_id)
    reports_dir.mkdir(parents=True,exist_ok=True)
    suffix = {"application/pdf":".pdf", "image/jpeg":".jpg", "image/png":".png"}[file.content_type]
    stored_path=reports_dir/f"{job_id}{suffix}"
    if not stored_path.exists():
        stored_path.write_bytes(content)
    with db.connect() as conn:
        conn.execute("INSERT INTO report_jobs(job_id,user_id,filename,content_type,byte_count,status,storage_path,content_sha256) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(job_id) DO NOTHING",
                     (job_id,user_id,Path(file.filename or "report").name,file.content_type,len(content),"RECEIVED_NOT_PROCESSED",str(stored_path),digest))
        row=conn.execute("SELECT * FROM report_jobs WHERE job_id=? AND user_id=?",(job_id,user_id)).fetchone()
    if row["status"] in {"RECEIVED_NOT_PROCESSED", "OCR_FAILED"} and settings.ocr_enabled:
        try:
            rally_catalog = json.loads((settings.normalized_data_dir / "rally.json").read_text(encoding="utf-8"))
            result = await extract_report_candidates(settings, content, file.content_type, rally_catalog["metrics"])
            transcription = result.get("transcription", "")
            staged=[]
            for candidate_data in result["candidates"]:
                candidate=ReportCandidate.model_validate(candidate_data)
                metric=next((item for item in rally_catalog["metrics"] if item["metric_code"]==candidate.metric_code),None)
                if metric and _report_candidate_has_evidence(candidate,metric,transcription):
                    staged.append(candidate)
            with db.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                current = conn.execute("SELECT status FROM report_jobs WHERE job_id=? AND user_id=?",(job_id,user_id)).fetchone()
                if current["status"] in {"RECEIVED_NOT_PROCESSED", "OCR_FAILED"} and staged:
                    conn.execute("UPDATE report_jobs SET candidates_json=?,status='AWAITING_USER_CONFIRMATION',updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND user_id=?",
                                 (json.dumps([item.model_dump(mode="json") for item in staged],ensure_ascii=False),job_id,user_id))
                elif current["status"] in {"RECEIVED_NOT_PROCESSED", "OCR_FAILED"}:
                    conn.execute("UPDATE report_jobs SET status='OCR_NO_CANDIDATES',updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND user_id=?",(job_id,user_id))
            with db.connect() as conn:
                row=conn.execute("SELECT * FROM report_jobs WHERE job_id=? AND user_id=?",(job_id,user_id)).fetchone()
            return {"job_id":job_id,"status":row["status"],"bytes":len(content),"sha256":digest,
                    "ocr_provider":result["provider"],"model":result["model"],"candidates":json.loads(row["candidates_json"]),
                    "unreadable":result["unreadable"],"usage":result["usage"],"transcription":transcription,
                    "message":"OCR candidates are unverified and await explicit user confirmation."}
        except Exception as error:
            with db.connect() as conn:
                conn.execute("UPDATE report_jobs SET status='OCR_FAILED',updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND user_id=? AND status IN ('RECEIVED_NOT_PROCESSED','OCR_FAILED')",(job_id,user_id))
            message = str(error) if isinstance(error, (RuntimeError, ValueError, FileNotFoundError, KeyError)) else "OCR provider request failed"
            raise HTTPException(status_code=502,detail={"code":"REPORT_OCR_FAILED","message":message})
    return {"job_id":job_id,"status":row["status"],"bytes":len(content),"sha256":digest,
            "ocr_provider":"openai-compatible" if settings.ocr_enabled else None,
            "ocr_model":settings.ocr_model if settings.ocr_enabled else None,
            "message":"OCR is disabled; no values were extracted." if not settings.ocr_enabled else "Report OCR is enabled."}


@router.get("/reports/extraction-jobs/{job_id}/file")
def report_file(job_id: str, user_id: str = Depends(get_user_id)):
    with db.connect() as conn:
        row = conn.execute("SELECT storage_path,filename,content_type FROM report_jobs WHERE job_id=? AND user_id=?",(job_id,user_id)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Report job not found")
    stored_path = Path(row["storage_path"] or "").resolve()
    user_report_dir = (settings.normalized_data_dir.parent / "report_uploads" / _report_user_storage_key(user_id)).resolve()
    if user_report_dir not in stored_path.parents or not stored_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(stored_path, media_type=row["content_type"], filename=Path(row["filename"]).name)


@router.get("/reports/extraction-jobs/{job_id}")
def report_job(job_id: str, user_id: str = Depends(get_user_id)) -> dict:
    with db.connect() as conn:
        row=conn.execute("SELECT * FROM report_jobs WHERE job_id=? AND user_id=?",(job_id,user_id)).fetchone()
    if row is None:
        raise HTTPException(status_code=404,detail="Report job not found")
    return {"job_id":row["job_id"],"status":row["status"],"filename":row["filename"],
            "sha256":row["content_sha256"],"candidates":json.loads(row["candidates_json"]),
            "confirmed_observations":json.loads(row["confirmed_json"]),
            "ocr_model":settings.ocr_model if row["status"] != "RECEIVED_NOT_PROCESSED" else None,
            "review_required":row["status"] == "AWAITING_USER_CONFIRMATION",
            "file_url":f"/v1/reports/extraction-jobs/{job_id}/file"}


@router.post("/reports/extraction-jobs/{job_id}/candidates")
def stage_report_candidates(job_id: str, payload: ReportCandidateConfirm, user_id: str = Depends(get_user_id)) -> dict:
    try:
        staged = [ReportCandidate.model_validate(candidate.model_dump(mode="json")) for candidate in payload.candidates]
        catalog=json.loads((settings.normalized_data_dir / "rally.json").read_text(encoding="utf-8"))["metrics"]
        catalog_by_code={metric["metric_code"]:metric for metric in catalog}
        for index,candidate in enumerate(staged):
            metric=catalog_by_code.get(candidate.metric_code)
            if metric is None or not _report_candidate_has_evidence(candidate,metric):
                raise HTTPException(status_code=422,detail={"code":"REPORT_CANDIDATE_UNVERIFIABLE","index":index})
    except ValidationError as error:
        raise HTTPException(status_code=422,detail={"code":"INVALID_CANDIDATES","message":str(error)})
    candidate_list=[candidate.model_dump(mode="json") for candidate in staged]
    candidate_json=json.dumps(candidate_list,ensure_ascii=False)
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM report_jobs WHERE job_id=? AND user_id=?", (job_id,user_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404,detail="Report job not found")
        if row["status"] not in {"RECEIVED_NOT_PROCESSED", "OCR_FAILED", "OCR_NO_CANDIDATES", "AWAITING_USER_CONFIRMATION"}:
            raise HTTPException(status_code=409,detail="Report job cannot accept staged candidates in its current state")
        conn.execute("UPDATE report_jobs SET candidates_json=?,status='AWAITING_USER_CONFIRMATION',updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND user_id=?",
                     (candidate_json,job_id,user_id))
    return {"job_id":job_id,"status":"AWAITING_USER_CONFIRMATION","candidates":candidate_list,
            "message":"Candidates are staged only; they are not observations until user confirmation."}


def _report_candidate_has_evidence(candidate: ReportCandidate, metric: dict, transcription: str | None = None) -> bool:
    from ..integrations.report_ocr import _is_known_analyte, _quote_supports_candidate

    return _is_known_analyte(candidate.source_analyte, metric) and _quote_supports_candidate(candidate, transcription)


@router.post("/reports/extraction-jobs/{job_id}/confirm")
def confirm_report_candidates(job_id: str, payload: ReportCandidateConfirm, user_id: str = Depends(get_user_id)) -> dict:
    created = []
    catalog = json.loads((settings.normalized_data_dir / "rally.json").read_text(encoding="utf-8"))["metrics"]
    catalog_by_code = {metric["metric_code"]: metric for metric in catalog}
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status,candidates_json FROM report_jobs WHERE job_id=? AND user_id=?", (job_id,user_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404,detail="Report job not found")
        if row["status"] != "AWAITING_USER_CONFIRMATION":
            raise HTTPException(status_code=409,detail="Report candidates are not awaiting confirmation")
        staged = json.loads(row["candidates_json"])
        submitted = [candidate.model_dump(mode="json") for candidate in payload.candidates]
        if submitted != staged:
            raise HTTPException(status_code=409,detail="Confirmed candidates must exactly match staged candidates")
        for index, item in enumerate(staged):
            candidate = ReportCandidate.model_validate(item)
            metric = catalog_by_code.get(candidate.metric_code)
            if metric is None or not _report_candidate_has_evidence(candidate, metric):
                raise HTTPException(status_code=422,detail={"code":"REPORT_CANDIDATE_UNVERIFIABLE","index":index})
            try:
                input_observation = candidate.model_dump(exclude={"source_analyte", "source_text", "source_page"})
                input_observation["provenance"] = {"source_type":"health_report","provider":"health_report",
                    "source_id":None,"sample_id":f"{job_id}:{index}","document_ref":job_id}
                input_observation["user_confirmed"] = True
                created.append(observations.add_in_transaction(conn,user_id,ObservationCreate(**input_observation)).model_dump(mode="json"))
            except (HTTPException,ValidationError,ValueError) as error:
                raise HTTPException(status_code=422,detail={"code":"REPORT_CANDIDATE_INVALID","index":index,"message":str(error)})
        ids = [item["observation_id"] for item in created]
        conn.execute("UPDATE report_jobs SET confirmed_json=?,status='CONFIRMED',updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND user_id=?",
                     (json.dumps(ids),job_id,user_id))
    return {"job_id":job_id,"status":"CONFIRMED","observations":created}
