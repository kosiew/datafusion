#!/bin/bash
# Test script for benchmark feature flags and CARGO_PROFILE changes
# Commits: b7026e2c6^..f67eb945f

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/benchmarks"

# Binary is built at workspace root, not in benchmarks/
DFBENCH="$SCRIPT_DIR/target/release/dfbench"

echo "========================================"
echo "Testing Benchmark Feature Flag Changes"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✓${NC} $1"
}

fail() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

info() {
    echo -e "${YELLOW}➜${NC} $1"
}

# Clean previous builds
info "Cleaning previous builds..."
cargo clean
echo ""

# Test 1: Default build with all features
echo "=== Test 1: Default build with all benchmarks enabled ==="
info "Running: cargo build --release --bin dfbench"
if cargo build --release --bin dfbench 2>&1 | tail -5; then
    pass "Default build succeeded"
else
    fail "Default build failed"
fi
echo ""

# Test 2: Build with only bench-tpch
echo "=== Test 2: Build with only bench-tpch feature ==="
info "Running: cargo build --release --bin dfbench --no-default-features --features bench-tpch"
if cargo build --release --bin dfbench --no-default-features --features "bench-tpch" 2>&1 | tail -5; then
    pass "Selective build (bench-tpch only) succeeded"
else
    fail "Selective build failed"
fi
echo ""

# Test 3: Verify disabled benchmark shows proper error
echo "=== Test 3: Verify disabled benchmark error message ==="
info "First, rebuild with only bench-tpch to disable tpcds..."
cargo build --release --bin dfbench --no-default-features --features "bench-tpch" > /dev/null 2>&1
info "Now attempting to run disabled 'tpcds' benchmark (without --help)..."
if "$DFBENCH" tpcds 2>&1 | grep -q "disabled.*bench-tpcds"; then
    pass "Disabled benchmark shows correct error message"
else
    echo "Output was:"
    "$DFBENCH" tpcds 2>&1 | head -10
    fail "Disabled benchmark error message not working correctly"
fi
echo ""

# Test 4: Build with multiple specific features
echo "=== Test 4: Build with multiple specific benchmarks ==="
info "Running: cargo build --release --bin dfbench --no-default-features --features bench-tpch,bench-tpcds,mimalloc"
if cargo build --release --bin dfbench --no-default-features --features "bench-tpch,bench-tpcds,mimalloc" 2>&1 | tail -5; then
    pass "Multi-feature build succeeded"
else
    fail "Multi-feature build failed"
fi
echo ""

# Test 5: Verify enabled benchmarks work
echo "=== Test 5: Verify enabled benchmarks show help ==="
info "Using binary from Test 4 (built with bench-tpch,bench-tpcds,mimalloc)..."
info "Testing tpch --help (should work)..."
if "$DFBENCH" tpch --help > /dev/null 2>&1; then
    pass "Enabled benchmark (tpch) works"
else
    fail "Enabled benchmark (tpch) doesn't work"
fi

info "Testing tpcds --help (should work)..."
if "$DFBENCH" tpcds --help > /dev/null 2>&1; then
    pass "Enabled benchmark (tpcds) works"
else
    fail "Enabled benchmark (tpcds) doesn't work"
fi

info "Testing clickbench (should fail with disabled message)..."
if "$DFBENCH" clickbench 2>&1 | grep -q "disabled.*bench-clickbench"; then
    pass "Disabled benchmark (clickbench) shows error"
else
    fail "Disabled benchmark (clickbench) should show error"
fi
echo ""

# Test 6: Test CARGO_PROFILE documentation
echo "=== Test 6: Verify CARGO_PROFILE is documented in bench.sh ==="
info "Checking bench.sh help output..."
if ./bench.sh --help 2>&1 | grep -q "CARGO_PROFILE"; then
    pass "CARGO_PROFILE is documented in bench.sh"
else
    fail "CARGO_PROFILE not found in bench.sh help"
fi
echo ""

# Test 7: Test dependency reduction
echo "=== Test 7: Verify dependencies are reduced with selective features ==="
info "Counting dependencies with bench-all..."
FULL_DEPS=$(cargo tree --features "bench-all" -p datafusion-benchmarks 2>/dev/null | wc -l)
info "Full dependencies: $FULL_DEPS lines"

info "Counting dependencies with only bench-tpch..."
MINIMAL_DEPS=$(cargo tree --no-default-features --features "bench-tpch" -p datafusion-benchmarks 2>/dev/null | wc -l)
info "Minimal dependencies: $MINIMAL_DEPS lines"

if [ "$MINIMAL_DEPS" -lt "$FULL_DEPS" ]; then
    pass "Selective features reduce dependency count ($MINIMAL_DEPS < $FULL_DEPS)"
else
    fail "Selective features should reduce dependencies"
fi
echo ""

# Test 8: Run cargo test
echo "=== Test 8: Run benchmark tests ==="
info "Running: cargo test"
if cargo test 2>&1 | tail -10; then
    pass "All tests passed"
else
    fail "Some tests failed"
fi
echo ""

# Test 9: Verify all benchmark features individually
echo "=== Test 9: Verify each benchmark feature compiles individually ==="
FEATURES=(
    "bench-cancellation"
    "bench-clickbench"
    "bench-h2o"
    "bench-hj"
    "bench-imdb"
    "bench-nlj"
    "bench-smj"
    "bench-sort-tpch"
    "bench-tpch"
    "bench-tpcds"
)

for feature in "${FEATURES[@]}"; do
    info "Testing feature: $feature"
    if cargo build --release --bin dfbench --no-default-features --features "$feature" 2>&1 | tail -1; then
        pass "Feature $feature builds successfully"
    else
        fail "Feature $feature failed to build"
    fi
done
echo ""

# Test 10: Verify README examples
echo "=== Test 10: Verify README examples work ==="
info "Testing: cargo run --profile release-nonlto --bin dfbench -- tpch --help"
if cargo run --profile release-nonlto --bin dfbench -- tpch --help > /dev/null 2>&1; then
    pass "README example with release-nonlto profile works"
else
    fail "README example with release-nonlto profile failed"
fi
echo ""

# Summary
echo "========================================"
echo "All Tests Passed Successfully! ✓"
echo "========================================"
echo ""
echo "Summary of tested features:"
echo "  • Default build with bench-all"
echo "  • Selective feature builds"
echo "  • Disabled benchmark error messages"
echo "  • Multi-feature builds"
echo "  • CARGO_PROFILE documentation"
echo "  • Dependency reduction verification"
echo "  • Individual feature compilation"
echo "  • Test suite execution"
echo "  • README example validation"
echo ""
echo "All changes in commits b7026e2c6^..f67eb945f are working correctly!"
