import hashlib
import hmac
import json
from fastapi.testclient import TestClient
from channels.app import create_webhook_app
from channels.webhook_idempotency import SqliteIdempotencyStore
from channels.whatsapp_metrics import AUDIO_RECEIVED, WhatsAppMetricsRecorder
from core.agent_executor import AgentExecutionResult, AgentExecutionStatus
SECRET = "p7b-secret"
class Sender:
    def __init__(self): self.sent = []
    def send_text(self, recipient, body): self.sent.append((recipient, body))
def payload(messages): return {"entry": [{"changes": [{"value": {"messages": messages}}]}]}
def signed(client, body):
    raw = json.dumps(body, separators=(",", ":")).encode()
    signature = "sha256=" + hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return client.post("/webhook/whatsapp", content=raw, headers={"Content-Type": "application/json", "X-Hub-Signature-256": signature})
def run(calls, fail=False):
    def executor(request):
        calls.append(request)
        if fail: raise RuntimeError("transient")
        return AgentExecutionResult(status=AgentExecutionStatus.COMPLETED, request_signature="s", correlation_id=request.correlation_id, output={"text": "ok"})
    return executor
def build(store, calls, sender, fail=False): return create_webhook_app(executor_fn=run(calls, fail), store=store, sender=sender, verify_token="token", app_secret=SECRET)
def message(wamid): return {"id": wamid, "from": "34600111222", "type": "text", "text": {"body": "hola"}}
def test_signature_rejects_missing_and_invalid(tmp_path):
    client = TestClient(build(SqliteIdempotencyStore(db_path=tmp_path / "q.db"), [], Sender()))
    assert client.post("/webhook/whatsapp", json=payload([message("a")])).status_code == 401
    assert client.post("/webhook/whatsapp", json=payload([message("a")]), headers={"X-Hub-Signature-256": "sha256=bad"}).status_code == 401
def test_signed_batch_processes_all_once(tmp_path):
    calls, sender = [], Sender(); client = TestClient(build(SqliteIdempotencyStore(db_path=tmp_path / "q.db"), calls, sender))
    body = {"entry": [{"changes": [{"value": {"messages": [message("m1"), message("m2")], "statuses": [{"id": "s1", "status": "delivered"}]}}, {"value": {"messages": [message("m3")], "statuses": [{"id": "s2", "status": "read"}]}}]}]}
    assert signed(client, body).status_code == 200; assert signed(client, body).status_code == 200
    assert len(calls) == 3 and len(sender.sent) == 3
def test_failed_job_recovers_after_restart(tmp_path):
    db = tmp_path / "q.db"; first_calls, sender = [], Sender()
    first = TestClient(build(SqliteIdempotencyStore(db_path=db), first_calls, sender, fail=True)); assert signed(first, payload([message("recover")])).status_code == 200
    recovered = []; second = TestClient(build(SqliteIdempotencyStore(db_path=db), recovered, sender)); assert signed(second, {"entry": []}).status_code == 200
    assert len(recovered) == 1

def test_durable_audio_records_received_metric(tmp_path):
    class Transcriber:
        def transcribe_media_id(self, media_id):
            return "hola"

    recorder, calls, sender = WhatsAppMetricsRecorder(), [], Sender()
    application = create_webhook_app(executor_fn=run(calls), store=SqliteIdempotencyStore(db_path=tmp_path / "q.db"), sender=sender, verify_token="token", app_secret=SECRET, transcriber=Transcriber(), recorder=recorder)
    client = TestClient(application)
    body = payload([{"id": "audio-1", "from": "34600111222", "type": "audio", "audio": {"id": "media-1"}}])
    assert signed(client, body).status_code == 200
    assert recorder.value(AUDIO_RECEIVED) == 1