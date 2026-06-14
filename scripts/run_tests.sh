#!/bin/bash
# Script to run all tests for the mooncake package
# Usage: ./scripts/run_tests.sh

set -e  # Exit immediately if a command exits with a non-zero status

# Ensure LD_LIBRARY_PATH includes /usr/local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# All tests below expect HTTP metadata at MC_METADATA_SERVER (default :8080).
# Start a standalone server if nothing is listening yet (bash /dev/tcp probe).
if ! (echo >/dev/tcp/127.0.0.1/8080) &>/dev/null; then
  echo "Starting mooncake_http_metadata_server on 127.0.0.1:8080..."
  mooncake_http_metadata_server --port 8080 &
  sleep 2
else
  echo "127.0.0.1:8080 already accepting connections; using existing metadata server."
fi

echo "Running transfer_engine tests..."
# Orphan targets (Ctrl+C, failed runs) reuse default ports and rpc_meta keys and break tests.
pkill -f "transfer_engine_target.py" 2>/dev/null || true
# Old targets DELETE rpc_meta on SIGTERM teardown. If a new target PUTs before an old
# process finishes, the old DELETE can remove the new key → initiator sees 404.
for _ in $(seq 1 30); do
  if ! pgrep -f "transfer_engine_target.py" >/dev/null 2>&1; then break; fi
  sleep 1
done
if pgrep -f "transfer_engine_target.py" >/dev/null 2>&1; then
  echo "WARNING: transfer_engine_target.py still running after pkill; sending SIGKILL..."
  pkill -9 -f "transfer_engine_target.py" 2>/dev/null || true
  sleep 1
fi
export MC_METADATA_SERVER="${MC_METADATA_SERVER:-http://127.0.0.1:8080/metadata}"
: "${TARGET_SERVER_NAME:=127.0.0.1:12345}"
: "${INITIATOR_SERVER_NAME:=127.0.0.1:12347}"
export TARGET_SERVER_NAME INITIATOR_SERVER_NAME
# HTTP metadata server rejects duplicate PUT for rpc_meta (400). Clear keys left when targets were SIGKILL'd.
python3 <<'PY'
import os, urllib.parse, urllib.error, urllib.request

def common_prefix():
    custom_key = ""
    p = os.environ.get("MC_METADATA_CLUSTER_ID")
    if p:
        custom_key = p
        if not custom_key.endswith("/"):
            custom_key += "/"
    return "mooncake/" + custom_key

def delete_key(meta, logical_key):
    url = meta + "?key=" + urllib.parse.quote(logical_key, safe="")
    req = urllib.request.Request(url, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as e:
        if e.code not in (200, 404):
            raise
    except Exception:
        pass

meta = os.environ.get("MC_METADATA_SERVER", "http://127.0.0.1:8080/metadata")
pre = common_prefix()
for hn in (
    os.environ.get("TARGET_SERVER_NAME", "127.0.0.1:12345"),
    os.environ.get("INITIATOR_SERVER_NAME", "127.0.0.1:12347"),
):
    for kind in ("rpc_meta", "ram"):
        delete_key(meta, pre + kind + "/" + hn)
PY

cd mooncake-wheel/tests
MC_FORCE_TCP=true python transfer_engine_target.py &
TARGET_PID=$!
# Wait until target publishes rpc_meta. Key must match TransferMetadata (incl. MC_METADATA_CLUSTER_ID)
# and URL encoding must match libcurl (curl_easy_escape). Require 2 consecutive OK reads so a late
# old-process DELETE cannot pass the wait then 404 the initiator.
TARGET_WAIT_SEC="${TARGET_METADATA_WAIT_SEC:-60}"
t=0
ok_streak=0
while (( t < TARGET_WAIT_SEC )); do
  if ! kill -0 "$TARGET_PID" 2>/dev/null; then
    echo "ERROR: transfer_engine_target.py (pid $TARGET_PID) exited before publishing metadata."
    echo "Inspect stderr above; common causes: metadata server down, init failure, or port conflict on TARGET_SERVER_NAME."
    exit 1
  fi
  if python3 -c "
import os, urllib.parse, urllib.request
custom_key = ''
p = os.environ.get('MC_METADATA_CLUSTER_ID')
if p:
    custom_key = p
    if not custom_key.endswith('/'):
        custom_key += '/'
logical_key = 'mooncake/' + custom_key + 'rpc_meta/' + os.environ.get('TARGET_SERVER_NAME', '127.0.0.1:12345')
meta = os.environ.get('MC_METADATA_SERVER', 'http://127.0.0.1:8080/metadata')
url = meta + '?key=' + urllib.parse.quote(logical_key, safe='')
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        raise SystemExit(0 if r.getcode() == 200 else 1)
except Exception:
    raise SystemExit(1)
"; then
    ok_streak=$((ok_streak + 1))
    if (( ok_streak >= 2 )); then
      break
    fi
  else
    ok_streak=0
  fi
  sleep 1
  t=$((t + 1))
done
if (( ok_streak < 2 )); then
  echo "ERROR: rpc_meta for ${TARGET_SERVER_NAME} not stable on ${MC_METADATA_SERVER} within ${TARGET_WAIT_SEC}s."
  echo "Logical key (same as TransferMetadata): mooncake/<MC_METADATA_CLUSTER_ID?>rpc_meta/${TARGET_SERVER_NAME}"
  kill "$TARGET_PID" 2>/dev/null || true
  exit 1
fi
MC_FORCE_TCP=true python transfer_engine_initiator_test.py
kill $TARGET_PID || true

echo "Running master tests..."

which mooncake_master 2>/dev/null | grep -q '/usr/local/bin/mooncake_master' && \
  { echo "ERROR: mooncake_master found in /usr/local/bin, not installed by python"; exit 1; } || \
  echo "mooncake_master not found in /usr/local/bin, installed by python"

echo "mooncake_master found, running tests..."
# Set a small kv lease ttl to make the test faster.
# Must be consistent with the client test parameters.
mooncake_master --default_kv_lease_ttl=500 &
MASTER_PID=$!
sleep 1
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_distributed_object_store.py
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_replicated_distributed_object_store.py
sleep 1
mooncake_client &
CLIENT_PID=$!
sleep 2
if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
  echo "ERROR: mooncake_client exited immediately (dummy-client tests need it listening on 127.0.0.1:50052)."
  echo "Typical cause: missing CUDA runtime (e.g. libcudart.so.13). Add CUDA lib64 to LD_LIBRARY_PATH, or use a wheel built for your installed CUDA."
  if CLIENT_BIN="$(command -v mooncake_client 2>/dev/null)"; then
    echo "mooncake_client: $CLIENT_BIN"
    ldd "$CLIENT_BIN" 2>/dev/null | grep -E "not found|libcudart" || ldd "$CLIENT_BIN" 2>/dev/null | head -n 20
  fi
  kill "$MASTER_PID" 2>/dev/null || true
  exit 1
fi
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_dummy_client.py
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_multi_dummy_clients.py --client-id client1 &
DUMMY_TEST_PID_1=$!
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_multi_dummy_clients.py --client-id client2 &
DUMMY_TEST_PID_2=$!
wait $DUMMY_TEST_PID_1 $DUMMY_TEST_PID_2
kill $CLIENT_PID || true

pip install numpy safetensors packaging
# Keep the test torch aligned with the EP/PG variants packaged into the CI wheel.
pip install "${MOONCAKE_TEST_TORCH_SPEC:-torch==2.11.0+cu128}" \
    --index-url "${MOONCAKE_TEST_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
    --extra-index-url https://pypi.org/simple
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_put_get_tensor.py
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_safetensor_functions.py
kill $MASTER_PID || true
wait $MASTER_PID 2>/dev/null || true


# Check if MOONCAKE_STORAGE_ROOT_DIR is set and not empty
if [ -n "$TEST_SSD_OFFLOAD_IN_EVICT" ]; then
    TEST_ROOT_DIR="/tmp/mooncake_test_ssd"
    mkdir -p $TEST_ROOT_DIR
    echo "MOONCAKE_STORAGE_ROOT_DIR is set to: $TEST_ROOT_DIR"
    echo "Running with ssd offload in evict tests..."
    # Set a small kv lease ttl to make the test faster.
    # Must be consistent with the client test parameters.
    mooncake_master --default_kv_lease_ttl=500 --root_fs_dir=$TEST_ROOT_DIR &
    MASTER_PID=$!
    sleep 1
    MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_ssd_offload_in_evict.py
    kill $MASTER_PID || true
    wait $MASTER_PID 2>/dev/null || true
    rm -rf $TEST_ROOT_DIR
else
    echo "Skipping test: MOONCAKE_STORAGE_ROOT_DIR environment variable is not set"
fi

if [ -n "$TEST_PROMOTION_ON_HIT" ]; then
    TEST_ROOT_DIR="/tmp/mooncake_test_promotion"
    mkdir -p $TEST_ROOT_DIR
    echo "Running L2->L1 promotion-on-hit e2e test..."
    # offload_on_evict drives the prerequisite SSD-only state; promotion_on_hit
    # turns the read path into a promotion trigger; threshold=1 makes the test
    # deterministic. --root_fs_dir is required so the master returns a non-
    # empty fsdir from GetStorageConfig, which is the trigger that initializes
    # the client's FileStorage (and therefore the offload heartbeat).
    mooncake_master \
        --default_kv_lease_ttl=500 \
        --root_fs_dir=$TEST_ROOT_DIR \
        --enable_offload=true \
        --offload_on_evict=true \
        --promotion_on_hit=true \
        --promotion_admission_threshold=1 &
    MASTER_PID=$!
    sleep 1
    # Lower bucket-flush thresholds so the test workload (~64 MB) actually
    # writes to disk rather than sitting in the bucket backend's ungrouped
    # pool until the default 500-key / 256-MB bucket fills.
    MC_METADATA_SERVER=http://127.0.0.1:8080/metadata \
        DEFAULT_KV_LEASE_TTL=500 \
        MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=$TEST_ROOT_DIR \
        MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=10 \
        MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES=10485760 \
        python test_promotion_on_hit.py
    kill $MASTER_PID || true
    wait $MASTER_PID 2>/dev/null || true
    rm -rf $TEST_ROOT_DIR
else
    echo "Skipping test: TEST_PROMOTION_ON_HIT environment variable is not set"
fi

echo "Running CXL protocol test (test_distributed_object_store_cxl.py)..."
killall mooncake_master || true
sleep 2

echo "Starting Mooncake Master with CXL enabled (--enable_cxl=true)..."
mooncake_master \
  --default_kv_lease_ttl=500 \
  --enable_cxl=true \
  &
CXL_MASTER_PID=$!
sleep 3
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_distributed_object_store_cxl.py
kill $CXL_MASTER_PID || true
wait $CXL_MASTER_PID 2>/dev/null || true
sleep 2
echo "CXL protocol test completed successfully!"

echo "Running CLI entry point tests..."
python test_cli.py

killall mooncake_http_metadata_server || true
killall mooncake_master || true
killall mooncake_client || true
mooncake_master --default_kv_lease_ttl=500 --enable_http_metadata_server=true &
MASTER_PID=$!
sleep 1
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 python test_distributed_object_store.py
sleep 1
kill $MASTER_PID || true
wait $MASTER_PID 2>/dev/null || true


echo "All tests completed successfully!"
cd ../..
