# LFS imputation method review

## Correct comparator

The relevant comparator is
[`PolicyEngine/young-worker-nics`](https://github.com/PolicyEngine/young-worker-nics),
not `nef-nics-reform`. Its NEET module uses five-quarter longitudinal LFS
panels as donors and PolicyEngine FRS people as receivers.

The method has two distinct parts:

1. a QRF supplies receiver-level variation by age, gender and employment
   income; and
2. the predictions are calibrated so their FRS-weighted mean equals the
   directly measured LFS entrant share.

It also reports an income-tercile estimator, calibrated to the same target, as
a sensitivity specification. Missing-pay donors are omitted from estimating
the earnings gradient but remain in the direct transition-rate target.

## Adaptation for this study

The available panel contains only 177 wave-1 manufacturing employees and 104
valid endpoint wage transitions. A forest fitted only to those observations
is not a credible primary model of detailed SIC heterogeneity. This study
therefore retains the transparent SIC-division × sex × age cells, with
credibility shrinkage and public BRES sector-mass alignment.

The key `young-worker-nics` calibration principle is now applied after
matching those cells to FRS:

- the FRS-weighted mean job-exit probability is calibrated to the direct,
  BRES-composition-adjusted LFS manufacturing exit rate;
- an income-tercile job-exit model is produced as a sensitivity estimate and
  calibrated to exactly the same rate; and
- receiver wage-change means are centred on the direct weighted LFS
  manufacturing wage-change estimate.

This separates *shape* from *level*: cells or earnings bands describe who has
higher predicted risk, while the longitudinal LFS determines the aggregate
transition level. The resulting outcomes remain imputed, not linked or
observed ASHE outcomes, and do not independently identify a causal tariff
effect.
