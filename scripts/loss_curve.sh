#!/bin/bash
# Binned mean training loss from a DiM-TS shard log. Single-batch losses at random
# diffusion timesteps are far too noisy to read one at a time -- the three-channel
# campaign looked like it was diverging until the readings were averaged in blocks.
#   bash scripts/loss_curve.sh <log> [bin_steps]
f=$1; bin=${2:-2000}
tr '\r' '\n' < "$f" \
| grep -oE "loss: [0-9.]+: +[0-9]+%\|[^|]*\| [0-9]+/[0-9]+" \
| sed -E 's/loss: ([0-9.]+).*\| ([0-9]+)\/[0-9]+/\2 \1/' \
| awk -v B="$bin" '{b=int($1/B); s[b]+=$2; n[b]++}
    END{printf "%-16s %10s %8s %9s\n","step","mean loss","n","vs prev";
        for(i=0;i<200;i++) if(n[i]>50){m=s[i]/n[i];
          d=(p>0)?sprintf("%+.1f%%",100*(m-p)/p):"-";
          printf "%6d-%-9d %10.5f %8d %9s\n", i*B,(i+1)*B,m,n[i],d; p=m}}'
