#!/usr/bin/env bash
# Run in your own Terminal — my sandbox can't reach these hosts.
set -e
cd "$(dirname "$0")/../docs/papers"
curl -L -o kelterer2023_pmgc.pdf \
  "https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2023/full_papers/66.pdf"
curl -L -o schuppler_turntaking_annotation.pdf \
  "https://arxiv.org/pdf/2504.09980"
ls -la
