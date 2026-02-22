#!/bin/bash
echo "Running smoke tests..."

HEALTH=$(curl -s http://localhost:8000/health)
echo "Health: $HEALTH"

if echo "$HEALTH" | grep -q "status"; then
  echo "Health check PASSED"
else
  echo "Health check FAILED"
  exit 1
fi

if curl -s http://localhost:8000/metrics | grep -q "request_count_total"; then
  echo "Metrics endpoint PASSED"
else
  echo "Metrics endpoint FAILED"
  exit 1
fi

echo "All smoke tests passed!"
