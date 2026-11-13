"""GRIDTECH data-center power portfolio optimiser, as a module for the website.

Built from Faraday.ipynb at the repo root (its single code cell), updated with the F1.ipynb changes (parallel per-year dispatch,
LP-relaxation screen, vectorised search/plots, fixed 12 MW battery, CO2 / capex / life-cycle cost by source): everything below the
header is the notebook's model, runner and plot code, unchanged except for the sizing solve: size() now uses the
two-step method described above it (method='full' restores the notebook's single MILP solve). The interactive questions
(Q1-Q10) are not here: team_app/gridtech_server.py takes those answers from the page and
validates them with the notebook's rules.
"""
from __future__ import annotations
import json, os, sys, time
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib
matplotlib.use('Agg')                # no window: the server saves the figures as PNG
import numpy as np
import scipy.sparse as sp

# SPEED / RULE settings (as in F1.ipynb) -----------------------------------------------
DISPATCH_WORKERS = min(10, os.cpu_count() or 1)   # model years solved at once (1 = serial)
BATTERY_MW = 12.0                                 # fixed 4-h battery size (MW) in every portfolio, for reliability. The website's
                                                  # battery switch-off / maximum / minimum still win (see Inputs._battery_rule).
                                                  # None = let the optimiser size the battery freely
BATTERY_TIER0_BACKUP = False                      # battery MW >= Tier-0 peak rule (ignored while BATTERY_MW is set)
DISPATCH_SOLVER = None                            # HiGHS LP method: None = default (simplex); try 'ipm' on large campuses
# ---------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]          # the repo: outputs/ are written here
# the contestant packet: data/ and starter/check_submission.py (v6.0.1 V2 data; GRIDTECH_PACKET=... to use another copy)
PACKET = Path(os.environ.get('GRIDTECH_PACKET') or ROOT / 'packet_v6.6.1').resolve()
sys.path.insert(0, str(PACKET / 'starter'))
import check_submission as ck       # packet's own input builder and accounting (unchanged)

OPT = ck.options.set_index('option_id')
RES = list(OPT.index)
OFFSITE = ['utility_solar', 'wind', 'nuclear']
GAS = ['gas_turbine', 'gas_engine', 'fuel_cell']
ETA = float(np.sqrt(OPT.loc['battery_4h', 'round_trip_efficiency']))
DUR = float(OPT.loc['battery_4h', 'duration_hours'])
BLK = {r: float(OPT.loc[r, 'block_mw']) for r in RES}
HPM = 730


@dataclass
class Reliability:
    onsite_n1: bool = False
    unit_mw: dict = field(default_factory=lambda: {'gas_turbine': 25.0, 'gas_engine': 5.0, 'fuel_cell': 5.0})
    grid_n1: bool = False
    grid_feed_survival: float = 0.0     # import capability left after losing one feed: 0 radial, 0.5 2x50%, 1.0 2x100%
    c2_protect: str = 'tier0'           # 'tier0' | 'tier0+tier1' | 'all'
    c2_allow_relocation: bool = True
    sb6_curtail_events: bool = False
    sb6_backup_fraction: float = 0.0

    def label(self):
        p = []
        if self.onsite_n1: p.append('N-1 onsite unit')
        if self.grid_n1: p.append(f'grid-loss N-1 (feed survival {self.grid_feed_survival:.0%}, protects {self.c2_protect})')
        if self.sb6_curtail_events: p.append('SB6 event curtailment (0 MW import)')
        if self.sb6_backup_fraction: p.append(f'SB6 backup >= {self.sb6_backup_fraction:.0%} of 75 MW')
        return '; '.join(p) or 'packet common rules'


@dataclass
class Custom:
    """Customer / project inputs. Leave a field as None to keep the GRIDTECH packet value.

    peak_mw          : mature campus peak demand (MW). Hourly shape and tier shares are kept,
                       scaled by peak_mw / 75.
    grid_schedule    : interconnection timeline as [(from_month, import_MW), ...] in development
                       months; every step must fall inside interconnection_window (months 0-24).
                       None = packet schedule (0/25/50/75 MW at months 0/18/24/30) scaled to peak.
    interconnection_month : shortcut for a single step: full import (= peak_mw) from this month (0-24).
    calendar_start   : 'YYYY-MM' calendar date of development month 0, used for labels only.
    earliest_start   : earliest month the campus can open (packet 14).
    land_acres       : onsite land for gas/battery equipment. None = 8 acres x peak/75.
    remote_capacity_mw: remote site for Tier 1 relocation. None = 12 MW x peak/75.
    scale_catalog    : True scales resource maxima (max_mw) by peak/75 so a bigger campus can be served.
    resources        : per-resource overrides, e.g.
                       {'gas_turbine': {'capex_usd_per_kw': 2200, 'available_month': 30},
                        'utility_solar': {'payment_usd_per_mwh': 45},
                        'nuclear': {'available_month': 72, 'max_mw': 50}}
                       Editable columns: capex_usd_per_kw, fom_usd_per_kw_year, life_years,
                       payment_usd_per_mwh, heat_rate_mmbtu_per_mwh_hhv, available_month, max_mw,
                       block_mw, land_acres_per_mw, round_trip_efficiency (battery).
    wholesale_growth / gas_growth / grid_co2_growth : real annual escalation (packet +5% / +3% / -3%).
    discount_rate    : real rate used in the capital recovery factor (packet 8%).
    stress_events    : replaces the packet's seven ERCOT events, e.g.
                       [{'id': 'heat_wave', 'start_utc': '2023-08-24T20:00:00Z', 'end_utc': '2023-08-25T03:00:00Z',
                         'import_fraction': 0.6, 'notice_minutes': 30}, ...]
                       Times fall in the reference year (2023) and repeat every model year. An hour is a stress hour if
                       an event covers any part of it. [] = no stress events. None = packet events.
    """
    peak_mw: float = 75.0
    grid_schedule: list | None = None
    interconnection_month: int | None = None      # shortcut: full import (= peak) from this month
    interconnection_window: tuple = (0, 24)       # allowed months for customer interconnection steps (any month up to 24)
    calendar_start: str | None = None
    earliest_start: int = 14
    land_acres: float | None = None
    remote_capacity_mw: float | None = None
    scale_catalog: bool = True
    resources: dict = field(default_factory=dict)
    wholesale_growth: float | None = None
    gas_growth: float | None = None
    grid_co2_growth: float | None = None
    discount_rate: float = 0.08
    stress_events: list | None = None
    tier2_max_fraction: float | None = None
    grid_adder_extra: float | None = None         # added (thesis scenario analysis): extra $/MWh on merchant electricity, e.g. new grid-cost rules       # added (thesis, example 4): share of Tier 2 that may be interrupted in a stress hour; None = packet value, 0 = no Tier 2 interruption

    def is_packet(self):
        d = Custom()
        same = {k: v for k, v in self.__dict__.items() if k not in ('calendar_start',)}
        ref = {k: v for k, v in d.__dict__.items() if k not in ('calendar_start',)}
        return same == ref


@dataclass
class Scenario:
    start: int = 20
    grid_delay: int = 0
    rel: Reliability = field(default_factory=Reliability)
    custom: Custom | None = None
    objective: str = 'cost'
    cost_cap: float | None = None
    co2_cap_kg_mwh: float | None = None
    forbid: tuple = ()
    max_caps: dict | None = None
    avail_override: dict | None = None
    min_caps: dict | None = None
    days_per_month: int = 2          # sample days per month for sizing (was 4; 4 = more accurate, slower)
    staged: bool = False            # True = capacity can be added each model year (never removed)
    eps_throughput: float = 0.01
    battery_tier0: bool | None = None   # battery MW >= Tier-0 peak; None = BATTERY_TIER0_BACKUP


REF_HOUR_UTC = np.array([np.datetime64(t.rstrip('Z')[:19]) for t in ck.hourly['timestamp_utc']], dtype='datetime64[m]')   # start of each reference hour


def stress_hours(events):
    """Customer stress events -> per reference hour (8760): stress active, import fraction, notice minutes.
    An hour is stressed if an event covers any part of it. Where two events touch the same hour, the stricter
    import fraction and the shorter notice apply."""
    act = np.zeros(REF_HOUR_UTC.size, bool); frac = np.ones(REF_HOUR_UTC.size); notice = np.zeros(REF_HOUR_UTC.size)
    end_of_hour = REF_HOUR_UTC + np.timedelta64(60, 'm')
    for e in events:
        s = np.datetime64(str(e['start_utc']).rstrip('Z')[:19], 'm'); t = np.datetime64(str(e['end_utc']).rstrip('Z')[:19], 'm')
        m = (REF_HOUR_UTC < t) & (end_of_hour > s)
        f, nm = float(e['import_fraction']), float(e.get('notice_minutes', 30))
        notice[m] = np.where(act[m], np.minimum(notice[m], nm), nm)
        frac[m] = np.minimum(frac[m], f)
        act |= m
    return act, frac, notice


class Inputs:
    """Full-horizon hourly arrays for one scenario: packet profiles + customer inputs (Custom)."""
    @staticmethod
    def _battery_rule(sc, opt):
        """How the battery is sized in this scenario (BATTERY_MW fixes it at 12 MW, but the website's inputs win):
        'fixed' = exactly `mw` in every year in service | 'off' = switched off on the website | 'max_override' = the website's
        maximum is below BATTERY_MW so the optimiser sizes it within that maximum | 'free' / 'tier0' = BATTERY_MW is None."""
        bt0 = BATTERY_TIER0_BACKUP if sc.battery_tier0 is None else sc.battery_tier0
        if BATTERY_MW is None:
            return dict(mode='tier0' if bt0 else 'free', mw=None, note=None)
        if 'battery_4h' in sc.forbid:
            return dict(mode='off', mw=0.0, note='Battery switched off on the website: no battery in this portfolio.')
        blk = float(opt.loc['battery_4h', 'block_mw'])
        want = float(np.ceil(float(BATTERY_MW) / blk - 1e-9) * blk)
        mn = float((sc.min_caps or {}).get('battery_4h', 0) or 0)
        user_max = (sc.max_caps or {}).get('battery_4h')
        if user_max is None and 'max_mw' in ((sc.custom.resources or {}).get('battery_4h', {}) if sc.custom else {}):
            user_max = opt.loc['battery_4h', 'max_mw']
        if user_max is None:                                   # catalogue maximum: raise it so the fixed battery fits
            opt.loc['battery_4h', 'max_mw'] = max(float(opt.loc['battery_4h', 'max_mw']), want, mn)
        elif np.floor(float(user_max) / blk + 1e-9) * blk < want - 1e-9:
            return dict(mode='max_override', mw=None, note=f'Battery maximum set to {float(user_max):g} MW on the website, below the '
                        f'fixed {BATTERY_MW:g} MW: the optimiser sizes the battery within your maximum.')
        tgt = float(np.ceil(max(want, mn) / blk - 1e-9) * blk)
        note = f'Battery fixed at {tgt:g} MW.' if tgt <= want + 1e-9 else \
            f'Battery fixed at {tgt:g} MW: your minimum of {mn:g} MW is above the standard {BATTERY_MW:g} MW.'
        return dict(mode='fixed', mw=tgt, note=note)

    def __init__(self, sc: Scenario):
        cu = sc.custom or Custom()
        c = ck.prepare_inputs({'initial_service_month': sc.start, 'grid_delay_months': sc.grid_delay, 'capacities_mw': {}})
        self.c = c; self.sc = sc; self.cu = cu; n = c['n']
        self.n = n; self.h0 = sc.start * HPM
        self.is_packet = cu.is_packet()
        k = cu.peak_mw / 75.0; self.k = k; self.peak = cu.peak_mw
        self.load, self.t0, self.t1, self.t2 = c['load'] * k, c['tier0'] * k, c['tier1'] * k, c['tier2'] * k
        self.month, self.year, self.active = c['month'], c['year'], c['active']
        ref = c['ref']; Hh = ck.hourly; yrs = np.arange(n) / 8760
        g = lambda col: Hh[col].to_numpy(float)[ref]
        tr = ck.trajectory['annual_real_rates']
        gw = tr['wholesale_energy'] if cu.wholesale_growth is None else cu.wholesale_growth
        gg = tr['henry_hub'] if cu.gas_growth is None else cu.gas_growth
        gc = tr['grid_co2'] if cu.grid_co2_growth is None else cu.grid_co2_growth
        self.price = g('wholesale_price_usd_per_mwh') * (1 + gw) ** yrs + g('grid_adder_usd_per_mwh') + float(cu.grid_adder_extra or 0.0)
        self.gas_price = g('henry_hub_usd_per_mmbtu') * (1 + gg) ** yrs + g('delivered_gas_adder_usd_per_mmbtu')
        mco2 = (1 + gc) ** yrs
        self.grid_co2 = g('grid_co2_t_per_mwh') * mco2
        self.remote_co2 = ck.response['remote_co2_t_per_mwh'] * mco2
        # interconnection timeline
        lo_w, hi_w = cu.interconnection_window
        if cu.grid_schedule is not None or cu.interconnection_month is not None:
            sched = list(cu.grid_schedule) if cu.grid_schedule is not None else [(int(cu.interconnection_month), cu.peak_mw)]
            bad = [(m_, mw_) for m_, mw_ in sched if mw_ > 0 and not (lo_w <= int(m_) <= hi_w)]
            if bad:
                raise ValueError(f'Interconnection steps must be between month {lo_w} and {hi_w}; got {bad}. '
                                 f'Example: grid_schedule=[(18, {cu.peak_mw / 2:g}), (24, {cu.peak_mw:g})] or interconnection_month=21')
            if max(mw_ for _, mw_ in sched) < cu.peak_mw - 1e-9:
                print(f'note: final import capacity {max(mw_ for _, mw_ in sched):g} MW is below peak load {cu.peak_mw:g} MW; '
                      'onsite generation must cover the gap')
            sched = [(0, 0.0)] + [(int(m_), float(mw_)) for m_, mw_ in sched]
        else:
            sched = [(int(r_.from_month), float(r_.max_import_mw) * k) for r_ in ck.grid_schedule.itertuples()]
        self.grid_schedule = sorted((int(m_), float(mw_)) for m_, mw_ in sched)
        ordinary = np.zeros(n)
        for fm, mw in self.grid_schedule:
            ordinary = np.where(self.month >= fm + (sc.grid_delay if mw > 0 else 0), mw, ordinary)
        self.cap_ord = ordinary
        if cu.stress_events is not None:            # customer's stress events replace the packet's
            act, frac, notice = stress_hours(cu.stress_events)
            self.active, frac, notice = act[ref], frac[ref], notice[ref]
        else:
            frac, notice = g('stress_import_fraction'), g('notice_minutes')
        self.cap_eff = ordinary * frac
        if sc.rel.sb6_curtail_events:
            self.cap_eff = np.where(self.active, 0.0, self.cap_eff)
        # demand response limits
        rp = ck.response
        self.remote = cu.remote_capacity_mw if cu.remote_capacity_mw is not None else rp['remote_capacity_mw'] * k
        eligible = self.active & (notice >= rp['relocation_notice_minutes'])
        self.reloc_lim = np.where(eligible, np.minimum(self.t1, self.remote / rp['remote_energy_multiplier']), 0.0)
        t2f = rp['tier2_max_fraction'] if cu.tier2_max_fraction is None else float(cu.tier2_max_fraction)
        self.curt_lim = np.where(self.active, self.t2 * t2f, 0.0)
        self.reloc_price, self.int_price = c['relocation_price'], c['interruption_price']
        # resource catalogue with customer overrides
        opt = OPT.copy()
        if cu.scale_catalog and k != 1:
            opt['max_mw'] = np.maximum(opt.block_mw, np.floor(opt.max_mw * k / opt.block_mw + 1e-9) * opt.block_mw)
        for rid, ov in (cu.resources or {}).items():
            if rid not in opt.index:
                raise ValueError(f'unknown resource {rid!r}; use one of {RES}')
            for col, v in ov.items():
                if col not in opt.columns:
                    raise ValueError(f'unknown column {col!r} for {rid}')
                opt.loc[rid, col] = v
            if 'heat_rate_mmbtu_per_mwh_hhv' in ov and 'co2_t_per_mwh' not in ov and rid in GAS:
                opt.loc[rid, 'co2_t_per_mwh'] = ov['heat_rate_mmbtu_per_mwh_hhv'] * 117 * 0.00045359237
        self.opt = opt
        self.blk = {r: float(opt.loc[r, 'block_mw']) for r in RES}
        self.battery_rule = self._battery_rule(sc, opt)
        self.eta = float(np.sqrt(opt.loc['battery_4h', 'round_trip_efficiency']))
        self.dur = float(opt.loc['battery_4h', 'duration_hours'])
        self.land = cu.land_acres if cu.land_acres is not None else ck.case['onsite_energy_land_acres'] * k
        self.mask, self.prof = {}, {}
        for r in RES:
            am = int((sc.avail_override or {}).get(r, opt.loc[r, 'available_month']))
            m = (self.month >= max(sc.start, am)).astype(float)
            col = opt.loc[r, 'profile_column']
            p = Hh[col].to_numpy(float)[ref] if col else np.ones(n)
            self.mask[r] = m; self.prof[r] = p * m
        self.fixed_per_mw = {}
        dr = cu.discount_rate
        for r in RES:
            row = opt.loc[r]
            crf = dr / (1 - (1 + dr) ** (-row.life_years))
            fixed = 1000 * (row.capex_usd_per_kw * crf + row.fom_usd_per_kw_year) * self.mask[r].sum() / 8760
            pay = row.payment_usd_per_mwh * self.prof[r].sum() if row.payment_basis == 'available_generation' else 0.0
            self.fixed_per_mw[r] = fixed + pay
        # SPEED: per-year sums with bincount instead of 70 boolean-mask scans of 87,600 hours
        self.fixed_per_mw_year = {}
        ny_ = n // 8760
        yidx = np.asarray(self.year).astype(int)
        for r in RES:
            row = opt.loc[r]
            crf = dr / (1 - (1 + dr) ** (-row.life_years))
            ann = 1000 * (row.capex_usd_per_kw * crf + row.fom_usd_per_kw_year)
            pay_rate = row.payment_usd_per_mwh if row.payment_basis == 'available_generation' else 0.0
            m_y = np.bincount(yidx, weights=self.mask[r], minlength=ny_)[:ny_]
            p_y = np.bincount(yidx, weights=self.prof[r], minlength=ny_)[:ny_]
            self.fixed_per_mw_year[r] = ann * m_y / 8760 + pay_rate * p_y
        self.flex = self.reloc_lim + self.curt_lim

    def sample(self, days):
        """Representative windows: `days` consecutive days mid-month for every month from
        opening, plus the full stress-event period. Returns hours, weights, segment ids."""
        hrs, w, seg = [], [], []
        sid = 0
        ref = self.c['ref']
        wind = ck.hourly['wind_proxy_cf'].to_numpy(float)[ref]; sol = ck.hourly['solar_proxy_cf'].to_numpy(float)[ref]
        L = 24 * days

        def best_window(a, b, exclude=None):
            """Window of L hours in [a,b) whose wind/solar/price/load means best match the
            month's means outside `exclude` (removes the sampling bias against renewables).
            SPEED: all candidate windows scored at once with cumulative sums."""
            nn = b - a
            keep = np.ones(nn, bool)
            if exclude is not None: keep[exclude[0] - a:exclude[1] - a] = False
            starts = np.arange(0, nn - L + 1, 24)
            if starts.size == 0:
                return None, keep.sum()
            nbad = np.r_[0, np.cumsum(~keep)]
            ok = (nbad[starts + L] - nbad[starts]) == 0
            err = np.zeros(starts.size)
            for s in (wind[a:b], sol[a:b], self.price[a:b], self.load[a:b]):
                t = s[keep].mean() if keep.any() else np.nan
                sc_ = max(abs(t), 1e-3)
                cs = np.r_[0.0, np.cumsum(s)]
                err += np.abs((cs[starts + L] - cs[starts]) / L - t) / sc_
            err[~ok] = np.inf
            if not np.isfinite(err).any():
                return None, keep.sum()
            return a + int(starts[int(np.argmin(err))]), keep.sum()

        def push(h, wt):
            nonlocal sid
            hrs.append(h); w.append(np.full(h.size, wt)); seg.append(np.full(h.size, sid)); sid += 1

        for m in range(self.sc.start, 120):
            a, b = m * HPM, (m + 1) * HPM
            act = np.nonzero(self.active[a:b])[0]
            if act.size:
                lo = max(a, a + act.min() - 12); hi = min(b, a + act.max() + 12)
                push(np.arange(lo, hi), 1.0)                  # stress period kept exactly
                s0, rest = best_window(a, b, (lo, hi))
                if s0 is not None and rest > 0:
                    push(np.arange(s0, s0 + L), rest / L)
            else:
                s0, _ = best_window(a, b)
                push(np.arange(s0, s0 + L), HPM / L)
        return np.concatenate(hrs), np.concatenate(w), np.concatenate(seg)


class LP:
    """Generic chronological model on a subset of hours.
    segments: SOC continuity within a segment; boundary 'cyclic' (sizing) or 'zero' (dispatch)."""
    def __init__(self, inp: Inputs, hours, w, seg, boundary='cyclic', fixed_caps=None, integer=True, soc_init=0.0, soc_end=0.0):
        self.inp, self.sc = inp, inp.sc
        self.hours, self.w, self.seg = hours, w, seg
        self.boundary, self.fixed, self.integer = boundary, fixed_caps, integer
        self.soc_init, self.soc_end = soc_init, soc_end
        self._build()

    def _build(self):
        I, sc, rel = self.inp, self.sc, self.sc.rel
        H, w, seg = self.hours, self.w, self.seg
        T = H.size; t = np.arange(T)
        names = ['grid'] + OFFSITE + GAS + ['ch', 'dis', 'soc', 'reloc', 'curt']
        self.off = {k: i * T for i, k in enumerate(names)}
        nh = len(names) * T
        NY = 10 if (sc.staged and self.fixed is None) else 1     # capacity decision per model year when staged
        self.NY = NY
        self.ci = {r: nh + i * NY + np.arange(NY) for i, r in enumerate(RES)}   # capacity block-count vars
        self.ni = {r: self.ci[r][-1] for r in RES}                               # final-year capacity
        base = nh + 7 * NY
        self.yi = {g: base + i for i, g in enumerate(GAS)}
        self.Li = base + 3
        nv = base + 4
        hy = I.year[H] if NY > 1 else np.zeros(T, int)
        capcol = lambda r_: self.ci[r_][hy]                                      # per-hour capacity column
        V = lambda k: self.off[k] + t
        lb = np.zeros(nv); ub = np.full(nv, np.inf); cost = np.zeros(nv); co2 = np.zeros(nv); integ = np.zeros(nv, int)

        maxc = {r: float((sc.max_caps or {}).get(r, I.opt.loc[r, 'max_mw'])) for r in RES}
        open_year = sc.start // 12
        for r in RES:
            for y, i in enumerate(self.ci[r]):
                ub[i] = 0 if r in sc.forbid else np.floor(maxc[r] / I.blk[r] + 1e-9)
                if NY > 1 and I.mask[r][I.year == y].sum() == 0:
                    ub[i] = 0                                   # nothing to commit before availability/opening
                elif NY == 1 or y >= max(open_year, int(I.opt.loc[r, 'available_month']) // 12):
                    lb[i] = np.ceil((sc.min_caps or {}).get(r, 0) / I.blk[r] - 1e-9)
                if self.fixed is not None:
                    lb[i] = ub[i] = self.fixed.get(r, 0) / I.blk[r]
                integ[i] = int(self.integer)
        # Battery backs up Tier 0: battery MW >= Tier-0 peak in every year it is in service
        if I.battery_rule['mode'] == 'fixed' and self.fixed is None:
            # Fixed battery: exactly the rule's MW (BATTERY_MW rounded up to whole blocks, or the website minimum if higher)
            b_ = 'battery_4h'; on_ = I.mask[b_] > 0
            nb_ = I.battery_rule['mw'] / I.blk[b_]
            for y, i in enumerate(self.ci[b_]):
                if NY > 1 and not (on_ & (I.year == y)).any():
                    continue                                   # not yet available / not open
                lb[i] = ub[i] = nb_
        bt0 = BATTERY_TIER0_BACKUP if sc.battery_tier0 is None else sc.battery_tier0
        if bt0 and I.battery_rule['mode'] == 'tier0' and self.fixed is None and 'battery_4h' not in sc.forbid:
            b_ = 'battery_4h'; on_ = I.mask[b_] > 0
            for y, i in enumerate(self.ci[b_]):
                sel = on_ & (I.year == y) if NY > 1 else on_
                if sel.any():
                    lb[i] = max(lb[i], np.ceil(I.t0[sel].max() / I.blk[b_] - 1e-9))
        for g in GAS:
            ub[self.yi[g]] = 1; integ[self.yi[g]] = int(self.integer)
        ub[self.Li] = max(rel.unit_mw.values())
        ub[V('reloc')] = I.reloc_lim[H]; ub[V('curt')] = I.curt_lim[H]

        cost[V('grid')] = w * I.price[H]; co2[V('grid')] = w * I.grid_co2[H]
        for g in GAS:
            cost[V(g)] = w * (I.opt.loc[g, 'payment_usd_per_mwh'] + I.opt.loc[g, 'heat_rate_mmbtu_per_mwh_hhv'] * I.gas_price[H])
            co2[V(g)] = w * I.opt.loc[g, 'co2_t_per_mwh']
        cost[V('reloc')] = w * I.reloc_price; cost[V('curt')] = w * I.int_price
        co2[V('reloc')] = w * 1.05 * I.remote_co2[H]
        cost[V('ch')] = cost[V('dis')] = w * sc.eps_throughput
        if self.fixed is None:
            for r in RES:
                if NY == 1:
                    cost[self.ci[r][0]] = I.blk[r] * I.fixed_per_mw[r]
                else:
                    cost[self.ci[r]] = I.blk[r] * I.fixed_per_mw_year[r]

        R, C, Vv, LO, HI = [], [], [], [], []
        nrow = [0]

        def add(blocks, lo, hi):
            m = max([np.size(lo), np.size(hi)] + [max(np.size(a), np.size(b)) for a, b in blocks])
            r0 = nrow[0] + np.arange(m)
            for ci, cv in blocks:
                ci = np.broadcast_to(ci, (m,)); cv = np.broadcast_to(np.asarray(cv, float), (m,))
                k = cv != 0
                R.append(r0[k]); C.append(ci[k]); Vv.append(cv[k])
            LO.append(np.broadcast_to(np.asarray(lo, float), (m,))); HI.append(np.broadcast_to(np.asarray(hi, float), (m,)))
            nrow[0] += m

        def add_row(ci, cv, lo, hi):
            R.append(np.full(len(ci), nrow[0])); C.append(np.asarray(ci)); Vv.append(np.asarray(cv, float))
            LO.append(np.array([lo], float)); HI.append(np.array([hi], float)); nrow[0] += 1

        one = np.ones(T)
        load = I.load[H]
        add([(V(k), one) for k in ['grid'] + OFFSITE + GAS] + [(V('dis'), one), (V('ch'), -one), (V('reloc'), one), (V('curt'), one)], load, load)
        add([(V(k), one) for k in ['grid'] + OFFSITE], -np.inf, I.cap_eff[H])
        for r in OFFSITE + GAS:
            add([(V(r), one), (capcol(r), -I.blk[r] * I.prof[r][H])], -np.inf, 0.0)
        nb = capcol('battery_4h'); bm = I.mask['battery_4h'][H]
        add([(V('ch'), one), (nb, -bm)], -np.inf, 0.0)
        add([(V('dis'), one), (nb, -bm)], -np.inf, 0.0)
        add([(V('soc'), one), (nb, -I.dur * bm)], -np.inf, 0.0)
        # SOC continuity within segments
        first = np.r_[True, seg[1:] != seg[:-1]]
        last = np.r_[seg[1:] != seg[:-1], True]
        if self.boundary == 'cyclic':
            seg_last = np.zeros(T, int)
            ends = np.nonzero(last)[0]; starts = np.nonzero(first)[0]
            for s0, e0 in zip(starts, ends): seg_last[s0] = e0
            prev = np.where(first, self.off['soc'] + seg_last, V('soc') - 1)
            pc = -one
        else:
            prev = np.where(first, V('soc'), V('soc') - 1)
            pc = np.where(first, 0.0, -1.0)
            ub[self.off['soc'] + np.nonzero(last)[0]] = self.soc_end
            lb[self.off['soc'] + np.nonzero(last)[0]] = self.soc_end
        rhs = np.where(first & (self.boundary != 'cyclic'), self.soc_init, 0.0)
        add([(V('soc'), one), (prev, pc), (V('ch'), -I.eta * one), (V('dis'), one / I.eta)], rhs, rhs)
        # V2 packet: the battery may charge from any source (grid, contracts or gas), so there is no clean-only charging row

        # annual cycle limit (weighted)
        yr = I.year[H]
        for y in np.unique(yr):
            m = yr == y
            lim = I.dur * 365 * I.mask['battery_4h'][I.year == y].sum() / 8760
            add_row(np.r_[V('dis')[m], self.ci['battery_4h'][y if NY > 1 else 0]], np.r_[w[m] / I.eta, -lim], -np.inf, 0.0)

        if self.fixed is None:
            # land
            add_row([self.ni[r] for r in RES], [I.blk[r] * I.opt.loc[r, 'land_acres_per_mw'] for r in RES], -np.inf, I.land)
            if NY > 1:   # staged: capacity never decreases
                for r in RES:
                    for y in range(1, NY):
                        add_row([self.ci[r][y], self.ci[r][y - 1]], [1.0, -1.0], 0.0, np.inf)
            # --- monthly adequacy rows cover EVERY hour (not only sampled ones) ---
            for mth in range(sc.start, 120):
                a, b = mth * HPM, (mth + 1) * HPM
                yy = (mth // 12) if NY > 1 else 0
                gas_cols = [self.ci[g][yy] for g in GAS]
                gm = [I.blk[g] * I.mask[g][a] for g in GAS]
                normal = np.max(I.load[a:b] - I.flex[a:b] - I.cap_eff[a:b])
                if normal > 0:
                    add_row(gas_cols, gm, normal, np.inf)
                if rel.onsite_n1:
                    need = np.max(I.load[a:b] - I.flex[a:b] - I.cap_eff[a:b])
                    add_row(gas_cols + [self.Li], gm + [-1.0], need, np.inf)
                if rel.grid_n1:
                    prot = {'tier0': I.t0, 'tier0+tier1': I.t0 + I.t1, 'all': I.load}[rel.c2_protect][a:b]
                    if rel.c2_allow_relocation and rel.c2_protect != 'tier0':
                        prot = prot - np.minimum(I.t1[a:b], I.remote / 1.05)
                    need = np.max(prot - rel.grid_feed_survival * I.cap_ord[a:b])
                    if need > 0:
                        add_row(gas_cols, gm, need, np.inf)
            if rel.onsite_n1:
                for g in GAS:
                    add_row([self.ni[g], self.yi[g]], [I.blk[g], -I.opt.loc[g, 'max_mw']], -np.inf, 0.0)
                    add_row([self.Li, self.yi[g]], [1.0, -float(rel.unit_mw[g])], 0.0, np.inf)
            if rel.sb6_backup_fraction > 0:
                for yy in (range(sc.start // 12, NY) if NY > 1 else [0]):
                    add_row([self.ci[g][yy] for g in GAS], [I.blk[g] for g in GAS], rel.sb6_backup_fraction * I.peak, np.inf)
            # optional caps (horizon totals, sample-weighted)
            if sc.co2_cap_kg_mwh is not None:
                k = sc.co2_cap_kg_mwh / 1000.0
                cols = np.r_[V('grid'), V('gas_turbine'), V('gas_engine'), V('fuel_cell'), V('reloc'), V('curt')]
                vals = np.r_[co2[V('grid')], co2[V('gas_turbine')], co2[V('gas_engine')], co2[V('fuel_cell')],
                             co2[V('reloc')] - k * 0.05 * w, k * w]
                add_row(cols, vals, -np.inf, k * float((w * load).sum()))
            if sc.cost_cap is not None:
                nz = np.nonzero(cost)[0]
                add_row(nz, cost[nz], -np.inf, sc.cost_cap)
        else:
            ub[self.Li] = 0
            for g in GAS: ub[self.yi[g]] = 0

        self.A = sp.csc_matrix((np.concatenate(Vv), (np.concatenate(R), np.concatenate(C))), shape=(nrow[0], nv))
        self.rlo, self.rhi = np.concatenate(LO), np.concatenate(HI)
        self.lb, self.ub, self.cost, self.co2, self.integ = lb, ub, cost, co2, integ
        # Dispatch (capacities fixed) is always least-cost economic dispatch.
        # In the CO2 objective, interrupted MWh are charged at the grid factor so that
        # dropping service is never counted as an emissions saving.
        if sc.objective == 'co2' and self.fixed is None:
            co2_obj = co2.copy(); co2_obj[V('curt')] += w * I.grid_co2[H]
            self.obj = co2_obj + 1e-7 * cost
        else:
            self.obj = cost

    def solve(self, backend='highs', time_limit=600, mip_gap=2e-2, verbose=False, start=None, lp_solver=None):
        """start: a feasible solution to begin the MIP search from (HiGHS only). lp_solver: HiGHS LP method for pure LPs."""
        t0 = time.time()
        self._start = start
        if backend == 'cplex':
            self.x, self.status = _solve_cplex(self, time_limit, mip_gap, verbose)
        else:
            self.x, self.status = _solve_highs(self, time_limit, mip_gap, verbose, lp_solver)
        self.seconds = time.time() - t0
        return self.status

    def caps(self):
        return {r: float(round(self.x[self.ni[r]]) * self.inp.blk[r]) for r in RES}

    def caps_by_year(self):
        """List of 10 dicts (model years 1..10). Same dict repeated when not staged."""
        out = []
        for y in range(10):
            yy = y if self.NY > 1 else 0
            out.append({r: float(round(self.x[self.ci[r][yy]]) * self.inp.blk[r]) for r in RES})
        return out

    def get(self, k):
        T = self.hours.size
        return np.clip(self.x[self.off[k]:self.off[k] + T], 0, None)


# ---------------------------------------------------------------- pipeline
# Two-step sizing (method='two-step', the default):
#   1. relax    the MILP with integers relaxed to continuous: an LP whose cost is a lower bound on every portfolio
#   2. restrict the MILP with each block count limited to the relaxed value +- SIZING['window'] blocks: a small integer search
#   3. polish   the full MILP, started from the step-2 portfolio, for at most SIZING['polish_s'] seconds; it can only improve on it
# Tested against the full MILP on 19 feasible cases: within 0.1% of its optimum where it finished (window 2), and about 24% cheaper
# on carbon-capped cases where it was still searching after 30 minutes. method='full' is the original single MILP solve.
SIZING = {'window': 2, 'polish_s': 120}
TWO_STEP_STATUS = 'Two-step: best found, not proven optimal'


def size(sc: Scenario, backend='highs', verbose=False, time_limit=600, method='two-step', on_step=None):
    """on_step(step, phase, info): optional progress callback; step 'relax' | 'restrict' | 'polish', phase 'start' | 'done'."""
    inp = Inputs(sc)
    H, w, seg = inp.sample(sc.days_per_month)
    lp = LP(inp, H, w, seg, boundary='cyclic')
    if method == 'full':
        st = lp.solve(backend, time_limit=time_limit, verbose=verbose)
        lp.steps = [dict(step='full', status=st, seconds=round(lp.seconds, 1), objective=float(lp.obj @ lp.x) if lp.x is not None else None)]
        return inp, lp, st
    note = on_step or (lambda *a: None)
    t0 = time.time()
    left = lambda: max(1.0, time_limit - (time.time() - t0))
    integ, lb0, ub0 = lp.integ.copy(), lp.lb.copy(), lp.ub.copy()
    ii = np.nonzero(integ)[0]
    steps = []

    def solve(step, budget, start=None):
        note(step, 'start', None)
        lp.solve(backend, time_limit=budget, verbose=verbose, start=start, lp_solver=DISPATCH_SOLVER)
        obj = float(lp.obj @ lp.x) if lp.x is not None else None
        steps.append(dict(step=step, status=lp.status, seconds=round(lp.seconds, 1), objective=obj,
                          bound=getattr(lp, 'mip_bound', None) if step == 'polish' else None))
        note(step, 'done', steps[-1])
        return lp.x, lp.status, obj

    lp.integ = np.zeros_like(integ)
    xr, st, _ = solve('relax', left())
    lp.integ = integ
    if xr is None:                      # relaxation infeasible: no portfolio can exist
        lp.x, lp.status, lp.seconds, lp.steps = None, st, time.time() - t0, steps
        return inp, lp, st
    k = SIZING['window']
    lp.lb[ii] = np.maximum(lb0[ii], np.floor(xr[ii] + 1e-6) - k)
    lp.ub[ii] = np.minimum(ub0[ii], np.ceil(xr[ii] - 1e-6) + k)
    x2, _, obj2 = solve('restrict', left())
    lp.lb, lp.ub = lb0, ub0
    x3, st3, obj3 = solve('polish', left() if x2 is None else min(left(), SIZING['polish_s']), start=x2)
    if x3 is not None and (obj2 is None or obj3 <= obj2 + 1e-6):
        x = x3
    else:
        x = x2
    st = 'Optimal' if st3 == 'Optimal' else (TWO_STEP_STATUS if x is not None else st3)
    lp.x, lp.status, lp.seconds, lp.steps = x, st, time.time() - t0, steps
    return inp, lp, st


# Carbon ceiling in the dispatch. Sizing meets the ceiling on the representative hours only, so the full 87,600-hour
# dispatch enforces it again, exactly, on the horizon totals the KPI uses:
#     combined CO2 <= cap x (local served MWh + remote MWh)
# The ten yearly LPs are linked only by that one row, so it is priced into every year's objective (cost + price x CO2 over
# the cap) and the price is found by bisection: the lowest price whose dispatch meets the ceiling. CO2_MARGIN keeps the
# result a hair under the ceiling so the export's round-off can never tip it over.
CO2_MARGIN = 1e-4
CO2_PRICE_MAX = 1e5             # $/t: beyond this the dispatch is effectively minimum-CO2
CO2_RESIZE_TRIES = 3            # re-sizings with a tighter internal ceiling when the sized portfolio cannot meet it


class CarbonCapUnreachable(RuntimeError):
    """The capacities cannot meet the carbon ceiling over the full horizon, even dispatched for minimum CO2."""
    def __init__(self, cap_kg, best_kg):
        self.cap_kg, self.best_kg = cap_kg, best_kg
        super().__init__(f'carbon ceiling {cap_kg:g} kg CO2/MWh not reachable with these capacities: '
                         f'the lowest-CO2 dispatch emits {best_kg:.1f} kg CO2/MWh over the 10 years')


def _pmap(fn, items, backend='highs', workers=None):
    """fn over items, in order, in up to `workers` threads (serial for workers <= 1, one item, or a non-HiGHS backend)."""
    from concurrent.futures import ThreadPoolExecutor
    nw = DISPATCH_WORKERS if workers is None else workers
    items = list(items)
    if backend != 'highs' or nw <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with ThreadPoolExecutor(max_workers=min(nw, len(items))) as ex:
        return list(ex.map(fn, items))


def dispatch(inp: Inputs, caps: dict, backend='highs', verbose=False, co2_cap_kg_mwh='scenario', info=None, workers=None):
    """Hourly LP over all hours, one model year at a time (SOC 0 at year boundaries).
    co2_cap_kg_mwh: horizon carbon ceiling enforced exactly ('scenario' = inp.sc.co2_cap_kg_mwh, None = no ceiling).
    Raises CarbonCapUnreachable when the capacities cannot meet it. info (dict) receives the carbon price and intensity."""
    import pandas as pd
    n, h0 = inp.n, inp.h0
    cap = inp.sc.co2_cap_kg_mwh if co2_cap_kg_mwh == 'scenario' else co2_cap_kg_mwh
    workers = DISPATCH_WORKERS if workers is None else workers
    cols = {k: np.zeros(n) for k in ['grid', 'reloc', 'curt', 'ch', 'dis', 'soc'] + OFFSITE + GAS}
    status, lps = [], []
    for y in range(10):
        a, b = max(h0, y * 8760), (y + 1) * 8760
        if a >= b: continue
        H = np.arange(a, b)
        cy = caps[y] if isinstance(caps, list) else caps
        lps.append(LP(inp, H, np.ones(H.size), np.zeros(H.size, int), boundary='zero', fixed_caps=cy, integer=False))
    if cap is None:
        # SPEED: the model years are independent LPs, solved in parallel threads (HiGHS releases the GIL while solving)
        def one(lp):
            st = lp.solve(backend, verbose=verbose, lp_solver=DISPATCH_SOLVER)
            if lp.x is None:
                raise RuntimeError(f'dispatch infeasible in model year {lp.hours[0] // 8760 + 1}: {st}')
            return st
        status = _pmap(one, lps, backend, workers)
        xs = [lp.x for lp in lps]
        price = 0.0
    else:
        xs, price, status = _dispatch_co2(inp, lps, cap, verbose, workers)
    for lp, x in zip(lps, xs):
        lp.x = x
        a, b = lp.hours[0], lp.hours[-1] + 1
        for k in cols: cols[k][a:b] = lp.get(k)
    d = pd.DataFrame({'model_hour': np.arange(n), 'grid_mw': cols['grid'], 'tier1_relocated_mw': cols['reloc'], 'tier2_curtailed_mw': cols['curt']})
    for r in RES:
        if r == 'battery_4h':
            d['battery_4h_charge_mw'] = cols['ch']; d['battery_4h_discharge_mw'] = cols['dis']; d['battery_4h_soc_mwh'] = cols['soc']
        else:
            d[r + '_mw'] = cols[r]
    d = _clean(d, inp, caps)
    if info is not None:
        info.update(co2_cap_kg_mwh=cap, co2_price_usd_per_t=price, co2_kg_mwh=_intensity(inp, d))
    return d, status


def _intensity(inp, d):
    """Horizon combined_electricity_co2_kg_per_mwh of a dispatch (the checker's formula)."""
    reloc = d.tier1_relocated_mw.to_numpy(float); curt = d.tier2_curtailed_mw.to_numpy(float)
    remote = reloc * ck.response['remote_energy_multiplier']
    co2 = d.grid_mw.to_numpy(float) @ inp.grid_co2 + remote @ inp.remote_co2
    co2 += sum(d[g + '_mw'].to_numpy(float).sum() * inp.opt.loc[g, 'co2_t_per_mwh'] for g in GAS)
    mwh = float((inp.load - reloc - curt).sum() + remote.sum())
    return 1000 * co2 / mwh if mwh > 0 else 0.0


def _dispatch_co2(inp, lps, cap_kg, verbose, workers=None):
    """Least-cost yearly dispatch meeting the horizon carbon ceiling. Returns (solutions, $/t carbon price, statuses)."""
    k = cap_kg / 1000 * (1 - CO2_MARGIN)
    rm = ck.response['remote_energy_multiplier']
    adj, const = [], 0.0
    for lp in lps:                      # per-variable CO2 minus k x (served + remote MWh); horizon sum <= 0 meets the ceiling
        T = lp.hours.size; V = lambda key: lp.off[key] + np.arange(T)
        a_ = lp.co2.copy()
        a_[V('curt')] += k                      # interrupted MWh leave the denominator
        a_[V('reloc')] -= k * (rm - 1)          # relocated MWh come back as rm x remote MWh
        adj.append(a_); const -= k * float(inp.load[lp.hours].sum())
    hs = [_highs_model(lp, verbose=verbose) for lp in lps]
    idx = [np.arange(lp.cost.size, dtype=np.int32) for lp in lps]

    def solve(lam):
        def one(z):                               # SPEED: the years are independent given the price: solved in parallel
            lp, h, a_, ix = z
            h.changeColsCost(ix.size, ix, lp.cost + lam * a_)
            h.run()
            st = h.modelStatusToString(h.getModelStatus()); sol = h.getSolution()
            if st != 'Optimal' or not sol.value_valid:
                raise RuntimeError(f'dispatch infeasible in model year {lp.hours[0] // 8760 + 1} at carbon price {lam:g} $/t: {st}')
            return np.array(sol.col_value), st
        out_ = _pmap(one, list(zip(lps, hs, adj, idx)), 'highs', workers)
        xs = [x for x, _ in out_]; sts = [st for _, st in out_]
        over = const + sum(float(a_ @ x) for a_, x in zip(adj, xs))
        return xs, sts, over                      # over <= 0: ceiling met

    served = sum(float(inp.load[lp.hours].sum()) for lp in lps)
    kg = lambda over: cap_kg * (1 - CO2_MARGIN) + 1000 * over / served     # approx. intensity, for the log only
    xs, sts, over = solve(0.0)
    if over <= 0:
        print(f'Carbon ceiling {cap_kg:g} kg CO2/MWh: met by the least-cost dispatch (~{kg(over):.1f} kg/MWh)')
        return xs, 0.0, sts
    print(f'Carbon ceiling {cap_kg:g} kg CO2/MWh: least-cost dispatch emits ~{kg(over):.1f}; pricing carbon to bring it under')
    hx, hs_, ho = solve(CO2_PRICE_MAX)
    if ho > 0:
        raise CarbonCapUnreachable(cap_kg, kg(ho))
    lo, hi = 0.0, 25.0
    while True:                                   # double until the ceiling is met
        x_, s_, o_ = solve(hi)
        if o_ <= 0:
            hx, hs_, ho = x_, s_, o_; break
        lo, hi = hi, hi * 2
        if hi >= CO2_PRICE_MAX:
            hi = CO2_PRICE_MAX; break
    for _ in range(30):                           # bisect to 0.1% on the price
        if hi - lo <= 1e-3 * hi: break
        mid = 0.5 * (lo + hi)
        x_, s_, o_ = solve(mid)
        if o_ <= 0: hi, hx, hs_, ho = mid, x_, s_, o_
        else: lo = mid
    print(f'Carbon ceiling {cap_kg:g} kg CO2/MWh: met at a carbon price of ${hi:,.1f}/t (~{kg(ho):.1f} kg/MWh)')
    return hx, hi, hs_


def _clean(d, inp, caps):
    """Remove solver noise so the export passes the checker's 1e-5 tolerances exactly."""
    ch = d.battery_4h_charge_mw.to_numpy().copy(); di = d.battery_4h_discharge_mw.to_numpy().copy()
    both = np.minimum(ch, di); ch -= both; di -= both            # net simultaneous flows
    ch[ch < 1e-7] = 0; di[di < 1e-7] = 0
    ETA = inp.eta
    soc = np.cumsum(ETA * ch - di / ETA)
    # year-boundary SOC is zero in the LP; drift is round-off only
    if abs(soc[-1]) > 1e-5 or soc.min() < -1e-5:
        print(f'note: SOC drift end={soc[-1]:.2e} min={soc.min():.2e}; correcting with last discharge')
    if soc[-1] > 0:
        j = np.nonzero(di)[0][-1] if di.any() else None
        if j is not None: di[j] += soc[-1] * ETA
    soc = np.cumsum(ETA * ch - di / ETA); soc[np.abs(soc) < 1e-9] = 0
    d['battery_4h_charge_mw'] = ch; d['battery_4h_discharge_mw'] = di; d['battery_4h_soc_mwh'] = np.clip(soc, 0, None)
    for c in d.columns[1:]:
        v = d[c].to_numpy().copy(); v[np.abs(v) < 1e-9] = 0; d[c] = v
    # exact balance: absorb residual round-off in merchant (tiny)
    local = inp.load - d.tier1_relocated_mw.to_numpy() - d.tier2_curtailed_mw.to_numpy()
    supply = d.grid_mw.to_numpy() + sum(d[r + '_mw'].to_numpy() for r in OFFSITE + GAS) + di - ch
    d['grid_mw'] = np.clip(d.grid_mw.to_numpy() + (local - supply), 0, None)
    # V2: the clean-sourced part of charging. Contract energy is attributed to the battery first, which is the most clean
    # credit allowed and always covers any contract energy above local demand (the checker requires both).
    d.insert(d.columns.get_loc('battery_4h_soc_mwh') + 1, 'battery_4h_clean_charge_mw',
             np.minimum(ch, sum(d[r + '_mw'].to_numpy() for r in OFFSITE)))
    return d


def _highs_model(m, time_limit=600, mip_gap=5e-3, verbose=False, lp_solver=None):
    """A HiGHS instance holding model m, not yet run."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue('output_flag', bool(verbose))
    h.setOptionValue('time_limit', float(time_limit))
    h.setOptionValue('mip_rel_gap', float(mip_gap))
    if lp_solver and not m.integ.any():
        h.setOptionValue('solver', lp_solver)       # optional: 'ipm' / 'simplex'
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = m.A.shape[1], m.A.shape[0]
    lp.col_cost_ = m.obj; lp.col_lower_ = m.lb; lp.col_upper_ = m.ub
    lp.row_lower_ = m.rlo; lp.row_upper_ = m.rhi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = m.A.indptr; lp.a_matrix_.index_ = m.A.indices; lp.a_matrix_.value_ = m.A.data
    if m.integ.any():
        K, Cc = highspy.HighsVarType.kInteger, highspy.HighsVarType.kContinuous
        lp.integrality_ = [K if v else Cc for v in m.integ]
    h.passModel(lp)
    return h


def _solve_highs(m, time_limit, mip_gap, verbose, lp_solver=None):
    import highspy
    h = _highs_model(m, time_limit, mip_gap, verbose, lp_solver)
    if getattr(m, '_start', None) is not None and m.integ.any():
        start = highspy.HighsSolution()
        start.col_value = list(m._start)
        start.value_valid = True
        h.setSolution(start)
    h.run()
    bd = h.getInfo().mip_dual_bound if m.integ.any() else None
    m.mip_bound = float(bd) if bd is not None and np.isfinite(bd) else None      # proven lower bound of the MIP searched
    st = h.modelStatusToString(h.getModelStatus())
    sol = h.getSolution()
    ok = st in ('Optimal', 'Time limit reached') and sol.value_valid
    return (np.array(sol.col_value) if ok else None), st


def _solve_cplex(m, time_limit, mip_gap, verbose):
    import cplex
    c = cplex.Cplex()
    if not verbose:
        c.set_log_stream(None); c.set_results_stream(None); c.set_warning_stream(None)
    c.parameters.timelimit.set(time_limit); c.parameters.mip.tolerances.mipgap.set(mip_gap)
    inf = cplex.infinity
    lb = np.where(np.isinf(m.lb), -inf, m.lb); ub = np.where(np.isinf(m.ub), inf, m.ub)
    if m.integ.any():
        c.variables.add(obj=m.obj.tolist(), lb=lb.tolist(), ub=ub.tolist(), types=''.join('I' if v else 'C' for v in m.integ))
    else:
        c.variables.add(obj=m.obj.tolist(), lb=lb.tolist(), ub=ub.tolist())
    lo = np.where(np.isinf(m.rlo), -inf, m.rlo); hi = np.where(np.isinf(m.rhi), inf, m.rhi)
    sen, rhs, rng = [], [], []
    for l, u in zip(lo, hi):
        if l == u: sen.append('E'); rhs.append(u); rng.append(0.0)
        elif l == -inf: sen.append('L'); rhs.append(u); rng.append(0.0)
        elif u == inf: sen.append('G'); rhs.append(l); rng.append(0.0)
        else: sen.append('R'); rhs.append(l); rng.append(u - l)
    # SPEED: add empty rows, then all coefficients in one call (no per-row SparsePair objects)
    c.linear_constraints.add(senses=sen, rhs=rhs, range_values=rng)
    A = m.A.tocoo()
    c.linear_constraints.set_coefficients(list(zip(A.row.tolist(), A.col.tolist(), A.data.tolist())))
    c.solve()
    st = c.solution.get_status_string()
    try:
        return np.array(c.solution.get_values()), st
    except Exception:
        return None, st


def score_custom(inp: Inputs, caps: dict, d):
    """Same accounting as starter/check_submission.py score(), on the customised inputs,
    plus the key physical checks. Used when inputs differ from the packet."""
    n = inp.n; tol = 1e-4
    cost = np.zeros(n); co2 = np.zeros(n); clean = np.zeros(n); imports = d.grid_mw.to_numpy(float).copy()
    supply = imports.copy(); ledger = []
    cby = caps if isinstance(caps, list) else [caps] * (n // 8760)
    caps = cby[-1]
    gen_ok = True
    for r in RES:
        row = inp.opt.loc[r]; mask = inp.mask[r]
        mw = np.array([cby[y].get(r, 0) for y in range(len(cby))])[inp.year]      # MW in force each hour
        dr = inp.cu.discount_rate; crf = dr / (1 - (1 + dr) ** (-row.life_years))
        fixed = mask * mw * 1000 * (row.capex_usd_per_kw * crf + row.fom_usd_per_kw_year) / 8760
        if row.kind == 'storage':
            gen = np.zeros(n)
            di = d.battery_4h_discharge_mw.to_numpy(float); ch = d.battery_4h_charge_mw.to_numpy(float)
            clean += ck.battery_clean_credit(ch, ck.clean_charge_column(d, r), di, inp.year); supply += di - ch
        else:
            gen = d[r + '_mw'].to_numpy(float); supply += gen
            gen_ok &= bool(np.max(gen - mw * inp.prof[r]) < tol)
            if row.location == 'offsite': imports += gen
            if row.clean_credit: clean += gen
        billed = mw * inp.prof[r] if row.payment_basis == 'available_generation' else gen
        pay = billed * row.payment_usd_per_mwh; fuel = gen * inp.gas_price * row.heat_rate_mmbtu_per_mwh_hhv
        cost += fixed + pay + fuel; co2 += gen * row.co2_t_per_mwh
        ledger.append({'option_id': r, 'selected_mw': float(mw.max()), 'capacity_charges_usd': float(fixed.sum()),
                       'energy_payment_or_variable_om_usd': float(pay.sum()), 'fuel_cost_usd': float(fuel.sum())})
    grid = d.grid_mw.to_numpy(float); cost += grid * inp.price; co2 += grid * inp.grid_co2
    reloc = d.tier1_relocated_mw.to_numpy(float); intr = d.tier2_curtailed_mw.to_numpy(float)
    remote = reloc * ck.response['remote_energy_multiplier']; local = inp.load - reloc - intr
    rcost = reloc * inp.reloc_price; icost = intr * inp.int_price; rco2 = remote * inp.remote_co2
    phys = {'balance_ok': bool(np.max(np.abs(supply - local)) < tol) and gen_ok,
            'import_cap_ok': bool(np.max(imports - inp.cap_eff) < tol),
            'tier0_served': bool(np.max(inp.t0 - local) < tol),
            'max_grid_import_mw': float(imports.max())}

    def agg(m):
        req = float(inp.load[m].sum()); srv = float(local[m].sum()); rem = float(remote[m].sum()); it = float(intr[m].sum())
        lc = float(cost[m].sum()); rr = float(rcost[m].sum()); ii = float(icost[m].sum()); ce = float(co2[m].sum()); re = float(rco2[m].sum())
        cm = float(clean[m].sum()); dv = lambda a, b: a / b if b > 0 else None
        return {'requested_mwh': req, 'local_served_mwh': srv, 'relocated_local_equivalent_mwh': float(reloc[m].sum()),
                'remote_electricity_mwh': rem, 'interrupted_mwh': it, 'local_energy_cost_usd': lc, 'relocation_cost_usd': rr,
                'interruption_consequence_usd': ii, 'total_decision_cost_usd': lc + rr + ii,
                'local_energy_cost_usd_per_mwh': dv(lc, srv), 'decision_cost_usd_per_requested_mwh': dv(lc + rr + ii, req),
                'local_operational_co2_t': ce, 'remote_operational_co2_t': re, 'combined_operational_co2_t': ce + re,
                'combined_electricity_co2_kg_per_mwh': dv(1000 * (ce + re), srv + rem), 'clean_energy_mwh': cm,
                'clean_energy_share': dv(cm, srv), 'merchant_mwh': float(grid[m].sum())}
    land = sum(caps.get(r, 0) * inp.opt.loc[r, 'land_acres_per_mw'] for r in RES)
    return {**phys, **agg(np.ones(n, bool)), 'land_acres': float(land), 'capacity_cost_ledger': ledger,
            'annual_results': [{'model_year': y + 1, **agg(inp.year == y)} for y in range(n // 8760)]}


def _f(x):
    return float(np.nan_to_num(float(x)))


def cost_breakdown(inp: Inputs, caps_by_year, d):
    """Capex and 10-year life-cycle cost by source (same formulas as the checker).
    upfront_capex_usd : gross capital of the capacity built (MW x $/kW x 1000), paid once.
    lifecycle_usd     : cost of the source over the model horizon = annualised capex (capital
                        recovery at the discount rate, hours in service only) + fixed O&M +
                        energy/contract payments + fuel; plus merchant energy and demand response.
    The life-cycle column sums to the 10-year decision cost."""
    import pandas as pd
    n = inp.n; ny = n // 8760; yr = np.asarray(inp.year).astype(int); dr = inp.cu.discount_rate
    cby = caps_by_year if isinstance(caps_by_year, list) else [caps_by_year] * ny
    z = dict(mw=0.0, upfront_capex_usd=0.0, annualised_capex_usd=0.0, fixed_om_usd=0.0,
             energy_payment_usd=0.0, fuel_usd=0.0, grid_energy_usd=0.0, demand_response_usd=0.0)
    rows = []
    for r in RES:
        row = inp.opt.loc[r]; mask = inp.mask[r]
        mw_y = np.array([cby[y].get(r, 0) for y in range(ny)], float); mw = mw_y[yr]
        crf = dr / (1 - (1 + dr) ** (-_f(row.life_years)))
        mwh_in_service = float((mask * mw).sum())
        if row.kind == 'storage':
            gen = np.zeros(n); delivered = float(d.battery_4h_discharge_mw.to_numpy(float).sum())
        else:
            gen = d[r + '_mw'].to_numpy(float); delivered = float(gen.sum())
        billed = mw * inp.prof[r] if row.payment_basis == 'available_generation' else gen
        added = np.clip(np.diff(np.r_[0.0, mw_y]), 0, None)
        rows.append({**z, 'source': BAR_LABELS[r], 'option_id': r, 'mw': float(mw_y.max()),
                     'upfront_capex_usd': float(added.sum()) * 1000 * _f(row.capex_usd_per_kw),
                     'annualised_capex_usd': mwh_in_service * 1000 * _f(row.capex_usd_per_kw) * crf / 8760,
                     'fixed_om_usd': mwh_in_service * 1000 * _f(row.fom_usd_per_kw_year) / 8760,
                     'energy_payment_usd': float((billed * _f(row.payment_usd_per_mwh)).sum()),
                     'fuel_usd': float((gen * inp.gas_price).sum()) * _f(row.heat_rate_mmbtu_per_mwh_hhv),
                     'mwh_delivered': delivered,
                     'capex_usd_per_kw': _f(row.capex_usd_per_kw), 'fom_usd_per_kw_year': _f(row.fom_usd_per_kw_year),
                     'payment_usd_per_mwh': _f(row.payment_usd_per_mwh), 'location': str(row.location)})
    grid = d.grid_mw.to_numpy(float)
    reloc = d.tier1_relocated_mw.to_numpy(float); intr = d.tier2_curtailed_mw.to_numpy(float)
    rows.append({**z, 'source': 'Merchant grid', 'option_id': 'grid',
                 'grid_energy_usd': float((grid * inp.price).sum()), 'mwh_delivered': float(grid.sum())})
    rows.append({**z, 'source': 'Tier 1 relocation', 'option_id': 'tier1_relocated',
                 'demand_response_usd': float((reloc * inp.reloc_price).sum()), 'mwh_delivered': float(reloc.sum())})
    rows.append({**z, 'source': 'Tier 2 interruption', 'option_id': 'tier2_curtailed',
                 'demand_response_usd': float((intr * inp.int_price).sum()), 'mwh_delivered': float(intr.sum())})
    df = pd.DataFrame(rows)
    df['lifecycle_usd'] = df[COST_PARTS].sum(axis=1)
    df['lifecycle_usd_per_mwh'] = [l / m if m > 0 else np.nan for l, m in zip(df.lifecycle_usd, df.mwh_delivered)]
    return df


COST_PARTS = ['annualised_capex_usd', 'fixed_om_usd', 'energy_payment_usd', 'fuel_usd', 'grid_energy_usd', 'demand_response_usd']
COST_PART_LABELS = {'annualised_capex_usd': 'Annualised capex', 'fixed_om_usd': 'Fixed O&M',
                    'energy_payment_usd': 'Energy / PPA payments', 'fuel_usd': 'Fuel',
                    'grid_energy_usd': 'Merchant grid energy', 'demand_response_usd': 'Relocation / interruption'}


def emissions_by_source(inp: Inputs, d):
    """Operational CO2 by source (tonnes): total over the horizon and by model year.
    Merchant grid at the hourly grid factor; gas units at their catalogue t/MWh; solar, wind and
    nuclear PPAs at their catalogue factor (zero operational); the battery itself emits nothing
    (charging energy is counted in the source that supplied it); Tier 1 relocation at the remote
    site (energy x multiplier) at the remote factor. Totals equal the combined operational CO2."""
    import pandas as pd
    ny = inp.n // 8760; yr = np.asarray(inp.year).astype(int)
    ser, mwh = {}, {}
    grid = d.grid_mw.to_numpy(float)
    ser['Merchant grid'] = grid * inp.grid_co2; mwh['Merchant grid'] = grid.sum()
    for r in OFFSITE + GAS:
        g = d[r + '_mw'].to_numpy(float)
        ser[BAR_LABELS[r]] = g * _f(inp.opt.loc[r, 'co2_t_per_mwh']); mwh[BAR_LABELS[r]] = g.sum()
    ser['4-h battery'] = np.zeros(inp.n); mwh['4-h battery'] = d.battery_4h_discharge_mw.to_numpy(float).sum()
    remote = d.tier1_relocated_mw.to_numpy(float) * ck.response['remote_energy_multiplier']
    ser['Tier 1 relocated (remote)'] = remote * inp.remote_co2; mwh['Tier 1 relocated (remote)'] = remote.sum()
    tot = sum(v.sum() for v in ser.values())
    df = pd.DataFrame([{'source': k, 'mwh': float(mwh[k]), 'co2_t': float(v.sum()),
                        'share': float(v.sum() / tot) if tot > 0 else 0.0,
                        'kg_co2_per_mwh': float(1000 * v.sum() / mwh[k]) if mwh[k] > 0 else 0.0} for k, v in ser.items()])
    ann = pd.DataFrame({k: np.bincount(yr, weights=v, minlength=ny)[:ny] for k, v in ser.items()},
                       index=pd.Index(range(1, ny + 1), name='model_year'))
    return df, ann


# ======================= RUNNER =======================
KEYS = ['total_decision_cost_usd', 'local_energy_cost_usd', 'relocation_cost_usd', 'interruption_consequence_usd',
        'decision_cost_usd_per_requested_mwh', 'local_energy_cost_usd_per_mwh', 'combined_electricity_co2_kg_per_mwh',
        'combined_operational_co2_t', 'clean_energy_share', 'merchant_mwh', 'interrupted_mwh',
        'relocated_local_equivalent_mwh', 'requested_mwh', 'land_acres', 'max_grid_import_mw']

# Reliability presets (cumulative). See model.py docstring for definitions.
PRESETS = {
    'R0': Reliability(),
    'R1': Reliability(onsite_n1=True),
    'R2': Reliability(onsite_n1=True, grid_n1=True, c2_protect='tier0'),
    'R3': Reliability(onsite_n1=True, grid_n1=True, c2_protect='tier0+tier1'),
    'R4': Reliability(onsite_n1=True, grid_n1=True, c2_protect='tier0+tier1', sb6_curtail_events=True, sb6_backup_fraction=0.5),
    'R5': Reliability(onsite_n1=True, grid_n1=True, c2_protect='all', sb6_curtail_events=True),
}
PRESET_TEXT = {
    'R0': 'Packet common rules only',
    'R1': '+ onsite N-1 (lose largest gas unit)',
    'R2': '+ loss of grid connection; Tier 0 served onsite',
    'R3': '+ loss of grid connection; Tier 0 + Tier 1 served (Tier 1 may relocate)',
    'R4': '+ Texas SB6: ERCOT cuts import to 0 MW in stress events; backup >= 50% of 75 MW',
    'R5': '+ full islanding of all load',
}


# ------------------------------------------------------------------ core
def run(sc: Scenario, out=None, backend='highs', verbose=False, caps=None, time_limit=600, sized=None):
    """Size (unless caps given) -> dispatch all 87,600 h -> score with packet checker -> export."""
    t0 = time.time()
    res = {}
    if caps is None:
        inp, lp, st = sized if sized else size(sc, backend, verbose, time_limit)
        res.update(sizing_status=st, sizing_s=round(lp.seconds, 1), sizing_steps=getattr(lp, 'steps', None))
        if lp.x is None:
            res['status'] = 'INFEASIBLE'
            return res, None
        caps = lp.caps()
        caps_y = lp.caps_by_year() if sc.staged else None
        res['sizing_objective'] = float(lp.obj @ lp.x)
    else:
        inp = Inputs(sc); caps_y = None
    cap = sc.co2_cap_kg_mwh
    co2 = {}
    sizing_cap = cap
    for attempt in range(CO2_RESIZE_TRIES + 1):
        try:
            t_d = time.time()
            d, dst = dispatch(inp, caps_y or caps, backend, co2_cap_kg_mwh=cap, info=co2)
            res['dispatch_s'] = round(res.get('dispatch_s', 0) + time.time() - t_d, 1)
            break
        except CarbonCapUnreachable as e:
            # sizing met the ceiling on the representative hours but the full horizon emits more: size again for a
            # proportionally tighter ceiling, then dispatch against the customer's real ceiling
            if not res.get('sizing_status') or attempt == CO2_RESIZE_TRIES:
                print(f'{e}. No portfolio is reported: it would break the carbon ceiling.')
                res.update(status='INFEASIBLE', sizing_status=f'Carbon ceiling not met over all 87,600 hours (best {e.best_kg:.1f} kg CO2/MWh)',
                           message=str(e), co2_best_kg_mwh=e.best_kg)
                return res, None
            sizing_cap = sizing_cap * min(0.97, cap / e.best_kg)
            print(f'{e}. Sizing again with an internal ceiling of {sizing_cap:.1f} kg/MWh (attempt {attempt + 2})')
            import dataclasses
            inp, lp, st = size(dataclasses.replace(sc, co2_cap_kg_mwh=sizing_cap), backend, verbose, time_limit)
            res.update(sizing_status=st, sizing_s=round(res['sizing_s'] + lp.seconds, 1))
            if lp.x is None:
                res.update(status='INFEASIBLE', message=f'carbon ceiling {cap:g} kg CO2/MWh: no portfolio when sized for {sizing_cap:.1f}')
                return res, None
            caps = lp.caps(); caps_y = lp.caps_by_year() if sc.staged else None
            res['sizing_objective'] = float(lp.obj @ lp.x)
    if cap is not None:
        res.update(co2_cap_kg_mwh=cap, co2_sizing_cap_kg_mwh=sizing_cap, co2_price_usd_per_t=co2['co2_price_usd_per_t'])
    res.update(reliability=sc.rel.label(), initial_service_month=sc.start, full_demand_month=sc.start + 10,
               grid_delay_months=sc.grid_delay, capacities=caps)
    res['capacities_by_year'] = caps_y or [caps] * 10
    res['staged'] = bool(caps_y)
    res['dispatch_status'] = sorted(set(dst))
    port = {'initial_service_month': sc.start, 'grid_delay_months': sc.grid_delay, 'capacities_mw': caps}
    acct = None
    try:
        if inp.is_packet and not caps_y:
            acct = ck.score(port, d)                 # official packet checker
            res['checker'] = 'PASS'
        else:
            acct = score_custom(inp, caps_y or caps, d)   # same accounting on customised / staged inputs
            ok = acct['balance_ok'] and acct['import_cap_ok'] and acct['tier0_served']
            res['checker'] = 'CUSTOM-OK' if ok else 'CUSTOM-FAIL'
        res.update({k: acct[k] for k in KEYS})
        if cap is not None:             # the scored intensity, from the same accounting as the checker
            res['co2_cap_met'] = bool(acct['combined_electricity_co2_kg_per_mwh'] <= cap)
            if not res['co2_cap_met']:
                raise RuntimeError(f"carbon ceiling {cap:g} kg CO2/MWh exceeded: {acct['combined_electricity_co2_kg_per_mwh']:.3f}")
        res['annual'] = [{k: a[k] for k in ['model_year', 'requested_mwh', 'total_decision_cost_usd',
                                            'combined_electricity_co2_kg_per_mwh', 'clean_energy_share', 'merchant_mwh']}
                         for a in acct['annual_results']]
    except ValueError as e:
        res['checker'] = f'FAIL: {e}'
    cdf = cost_breakdown(inp, caps_y or caps, d)
    edf, eann = emissions_by_source(inp, d)
    res['_cost_df'], res['_co2_df'], res['_co2_annual'] = cdf, edf, eann
    res['cost_by_source'] = cdf.to_dict('records')
    res['co2_by_source'] = edf.to_dict('records')
    res['co2_by_source_annual_t'] = eann.reset_index().to_dict('records')
    res['upfront_capex_usd'] = float(cdf.upfront_capex_usd.sum())
    bt0 = BATTERY_TIER0_BACKUP if sc.battery_tier0 is None else sc.battery_tier0
    br = inp.battery_rule
    res['battery_tier0_backup'] = br['mode'] == 'tier0'
    res['battery_fixed_mw'] = br['mw'] if br['mode'] in ('fixed', 'off') else None
    res['battery_rule'] = br['mode']
    res['battery_note'] = br['note']
    res['status'] = 'OK'
    res['total_s'] = round(time.time() - t0, 1)
    res['peak_mw'] = inp.peak; res['grid_schedule'] = inp.grid_schedule; res['calendar_start'] = inp.cu.calendar_start
    res['_inp'] = inp
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        (out / 'portfolio.json').write_text(json.dumps(port, indent=1))
        d.to_csv(out / 'dispatch.csv', index=False, float_format='%.10g')
        if acct:
            (out / ('check_results.json' if (inp.is_packet and not caps_y) else 'custom_results.json')).write_text(json.dumps(acct, indent=1))
        (out / 'summary.json').write_text(json.dumps({k: v for k, v in res.items() if not k.startswith('_')}, indent=1, default=float))
        cdf.to_csv(out / 'cost_by_source.csv', index=False)
        edf.to_csv(out / 'co2_by_source.csv', index=False)
        eann.to_csv(out / 'co2_by_source_annual_t.csv')
    return res, d


def month_to_date(m, cal):
    """Development month -> 'Mon YYYY' given calendar_start 'YYYY-MM' for month 0."""
    if not cal:
        return f'm{m}'
    y, mo = map(int, cal.split('-')[:2]); t = (mo - 1) + int(m)
    return f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][t % 12]} {y + t // 12}"


def show_inputs(custom=None, start=20):
    """Print the effective inputs (after customer overrides) - use before running."""
    import pandas as pd
    inp = Inputs(Scenario(start=start, custom=custom))
    print(f'Peak load {inp.peak:g} MW | annual energy at full demand {inp.load[-8760:].sum() / 1e3:,.0f} GWh | '
          f'land {inp.land:g} acres | remote relocation site {inp.remote:g} MW')
    print('Interconnection schedule:', ', '.join(f'{month_to_date(m_, inp.cu.calendar_start)}: {mw_:g} MW' for m_, mw_ in inp.grid_schedule))
    on_ = inp.mask['battery_4h'] > 0
    t0p = float(inp.t0[on_].max()) if on_.any() else 0.0
    bb = inp.blk['battery_4h']
    if inp.battery_rule['mode'] == 'fixed':
        bfix = inp.battery_rule['mw']
        print(f'Battery fixed at {bfix:g} MW / {bfix * inp.dur:g} MWh from {month_to_date(max(start, int(inp.opt.loc["battery_4h", "available_month"])), inp.cu.calendar_start)} '
              f'(Tier-0 peak {t0p:.1f} MW)')
    else:
        need = np.ceil(t0p / bb - 1e-9) * bb; bmax = float(inp.opt.loc['battery_4h', 'max_mw'])
        print(f'Tier-0 peak {t0p:.1f} MW -> battery backup needs >= {need:g} MW (battery max {bmax:g} MW)'
              + ('   WARNING: above battery max - raise battery max_mw or the run will be infeasible' if need > bmax + 1e-9 else ''))
    cols = ['available_month', 'block_mw', 'max_mw', 'capex_usd_per_kw', 'fom_usd_per_kw_year', 'life_years',
            'payment_usd_per_mwh', 'heat_rate_mmbtu_per_mwh_hhv', 'co2_t_per_mwh', 'land_acres_per_mw']
    t = inp.opt[cols].copy()
    t['$/MW-yr fixed'] = [inp.fixed_per_mw[r] / max(inp.mask[r].sum(), 1) * 8760 for r in t.index]
    print(t.round(3).to_string())
    return t


def line(tag, r):
    if r.get('status') != 'OK':
        return f'{tag:<12} {r.get("status")} ({r.get("sizing_status")})'
    c = r['capacities']
    mix = ', '.join(f'{k} {v:g}' for k, v in c.items() if v)
    cal = r.get('calendar_start')
    when = (f'open {month_to_date(r["initial_service_month"], cal)} full {month_to_date(r["full_demand_month"], cal)} ' if cal else '')
    return (f'{tag:<12} {when}open m{r["initial_service_month"]:<3} full m{r["full_demand_month"]:<3} '
            f'${r["total_decision_cost_usd"] / 1e6:7.1f}M  ${r["decision_cost_usd_per_requested_mwh"]:5.1f}/MWh  '
            f'{r["combined_electricity_co2_kg_per_mwh"]:5.0f} kg/MWh  clean {r["clean_energy_share"]:.0%}  '
            f'interrupted {r["interrupted_mwh"]:,.0f} MWh  [{r["checker"]}]  | {mix}')


# ------------------------------------------------------------------ modes
def ladder(start=20, presets=None, grid_delay=0, backend='highs', verbose=False, out=None, custom=None):
    from types import SimpleNamespace
    a = SimpleNamespace(start=start, presets=presets or list(PRESETS), grid_delay=grid_delay, backend=backend, verbose=verbose, out=out)
    rows = []
    for k in a.presets:
        sc = Scenario(start=a.start, grid_delay=a.grid_delay, rel=PRESETS[k], custom=custom)
        r, _ = run(sc, out=f'{a.out or "outputs/ladder"}/{k}', backend=a.backend, verbose=a.verbose)
        print(line(k, r), flush=True)
        rows.append((k, r))
    base = rows[0][1].get('total_decision_cost_usd')
    print('\nCost of reliability (vs first preset):')
    for k, r in rows:
        if r.get('status') == 'OK' and base:
            print(f'  {k}: {PRESET_TEXT[k]:<80} {r["total_decision_cost_usd"] / base - 1:+.1%}')


def lexico(reliability='R0', deadline=None, speed_slack=0, green_tolerance=0.05,
           co2_cap=None, grid_delay=0, backend='highs', verbose=False, out=None, custom=None):
    """Phase 1 speed -> Phase 2 cost -> Phase 3 green, each with an explicit tolerance."""
    from types import SimpleNamespace
    a = SimpleNamespace(reliability=reliability, deadline=deadline, speed_slack=speed_slack, green_tolerance=green_tolerance,
                        co2_cap=co2_cap, grid_delay=grid_delay, backend=backend, verbose=verbose, out=out)
    rel = PRESETS[a.reliability]
    out = Path(a.out or f'outputs/lexico_{a.reliability}')
    log = []
    cache = {}                                  # SPEED: reuse Phase-1 sizing in Phase 2

    # Phase 1: earliest feasible opening month
    print('Phase 1 - speed to power: searching earliest feasible opening month ...', flush=True)
    s_star, tested = None, []
    last = (a.deadline - 10) if a.deadline else 109
    for s in range((custom or Custom()).earliest_start, last + 1):
        sc = Scenario(start=s, grid_delay=a.grid_delay, rel=rel, custom=custom)
        infeas, st_r = relaxation_infeasible(sc, a.backend)     # SPEED: quick LP screen
        if infeas:
            tested.append((s, f'LP relaxation {st_r}'))
            print(f'  opening month {s}: {st_r} (LP relaxation)', flush=True)
            continue
        inp, lp, st = size(sc, a.backend, a.verbose)
        tested.append((s, st))
        print(f'  opening month {s}: {st}', flush=True)
        if lp.x is not None:
            s_star = s
            cache[s] = (inp, lp, st)
            break
    if s_star is None:
        print(f'No feasible opening found in months 14..{last}. Relax the deadline or the reliability standard '
              f'(this is a search result, not proof of impossibility).')
        return
    log.append(f'Phase 1: earliest feasible opening = month {s_star} (full demand month {s_star + 10}); tested {tested}')

    # Phase 2: lowest cost per requested MWh within [s*, s* + slack] (and the hard deadline)
    hi = s_star + a.speed_slack
    if a.deadline:
        hi = min(hi, a.deadline - 10)
    print(f'Phase 2 - cost: comparing openings {s_star}..{hi} on $/requested MWh ...', flush=True)
    cands = []
    for s in range(s_star, hi + 1):
        sc = Scenario(start=s, grid_delay=a.grid_delay, rel=rel, custom=custom)
        sized = cache.get(s) or size(sc, a.backend, a.verbose)
        if sized[1].x is None:
            continue
        r, _ = run(sc, out=out / f'phase2_m{s}', backend=a.backend, sized=sized)
        print('  ' + line(f'm{s}', r), flush=True)
        cands.append((r['decision_cost_usd_per_requested_mwh'], s, r, sized))
    _, s2, r2, sized2 = min(cands, key=lambda x: x[0])
    log.append(f'Phase 2: cost-optimal opening month {s2}: ' + line('', r2))

    # Phase 3: minimise CO2 at that opening, cost <= (1 + tolerance) x Phase-2 cost
    cap = (1 + a.green_tolerance) * sized2[1].obj @ sized2[1].x
    print(f'Phase 3 - green: minimise CO2 with cost <= +{a.green_tolerance:.0%} of Phase 2 ...', flush=True)
    sc3 = Scenario(start=s2, grid_delay=a.grid_delay, rel=rel, custom=custom, objective='co2', cost_cap=float(cap), co2_cap_kg_mwh=a.co2_cap)
    r3, _ = run(sc3, out=out / 'phase3_recommended', backend=a.backend, verbose=a.verbose)
    print('  ' + line('green', r3), flush=True)
    log.append('Phase 3: ' + line('', r3))

    print('\nRECOMMENDATION (' + PRESET_TEXT[a.reliability] + ')')
    print('\n'.join(log))
    if r3.get('status') == 'OK':
        dc = r3['total_decision_cost_usd'] / r2['total_decision_cost_usd'] - 1
        dco2 = r3['combined_electricity_co2_kg_per_mwh'] - r2['combined_electricity_co2_kg_per_mwh']
        print(f'Green step: cost {dc:+.1%}, intensity {dco2:+.0f} kg CO2/MWh vs cost-optimal. Files: {out / "phase3_recommended"}')
    (out / 'recommendation.txt').write_text('\n'.join(log) + '\n')



# ======================= PLOTS =======================
COLORS = {'nuclear': '#111111', 'wind': '#2e7d4f', 'utility_solar': '#f7c08a', 'battery_4h': '#8cc9a1',   # Powerllel palette: orange, black, grey, green
          'gas_turbine': '#8a3b0e', 'gas_engine': '#f26b1d', 'fuel_cell': '#d9822b', 'grid': '#9a9a96',
          'tier1_relocated': '#5f6b63', 'tier2_curtailed': '#000000'}
LABELS = {'nuclear': 'Nuclear PPA', 'wind': 'Wind PPA', 'utility_solar': 'Solar PPA', 'battery_4h': 'Battery discharge',
          'gas_turbine': 'Gas turbine', 'gas_engine': 'Gas engine', 'fuel_cell': 'Fuel cell', 'grid': 'Merchant grid',
          'tier1_relocated': 'Tier 1 relocated', 'tier2_curtailed': 'Tier 2 interrupted'}
BAR_LABELS = {'nuclear': 'Nuclear PPA', 'wind': 'Wind PPA', 'utility_solar': 'Solar PPA', 'battery_4h': '4-h battery',
              'gas_turbine': 'Gas turbine', 'gas_engine': 'Gas engine', 'fuel_cell': 'Fuel cell'}
STACK = ['nuclear', 'wind', 'utility_solar', 'gas_turbine', 'gas_engine', 'fuel_cell', 'battery_4h', 'grid',
         'tier1_relocated', 'tier2_curtailed']


def _supply(d):
    s = {k: d[k + '_mw'].to_numpy(float) for k in ['nuclear', 'wind', 'utility_solar', 'gas_turbine', 'gas_engine', 'fuel_cell']}
    net_batt = d.battery_4h_discharge_mw.to_numpy(float) - d.battery_4h_charge_mw.to_numpy(float)
    s['battery_4h'] = np.clip(net_batt, 0, None)
    # battery charging: remove it from contract supply first (as the clean-charge column attributes it), then grid, then gas
    ch = np.clip(-net_batt, 0, None)
    s['grid'] = d.grid_mw.to_numpy(float)
    for k in ['utility_solar', 'wind', 'nuclear', 'grid', 'gas_turbine', 'gas_engine', 'fuel_cell']:
        take = np.minimum(ch, s[k]); s[k] = s[k] - take; ch = ch - take
    s['tier1_relocated'] = d.tier1_relocated_mw.to_numpy(float)
    s['tier2_curtailed'] = d.tier2_curtailed_mw.to_numpy(float)
    return s


def _monthly_max(x):
    """SPEED: max per 730-hour model month via reshape (replaces 120 boolean-mask scans)."""
    return np.asarray(x, float)[:120 * HPM].reshape(120, HPM).max(axis=1)


def plot_results(r, d, title=None, save=True, show=True, inp=None):
    """Dashboard: (1) 10-year timeline monthly mix vs demand and grid limit with milestones,
    (2) portfolio capacities, (3) a typical mature week, (4) a stress-event window,
    (5) annual cost and CO2 intensity."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    start, delay = r['initial_service_month'], r.get('grid_delay_months', 0)
    inp = inp or r.get('_inp') or Inputs(Scenario(start=start, grid_delay=delay))
    load, ordinary, eff, active = inp.load, inp.cap_ord, inp.cap_eff, inp.active
    cal = inp.cu.calendar_start
    md = lambda m_: month_to_date(m_, cal)
    s = _supply(d)
    used = [k for k in STACK if s[k].sum() > 1e-6]
    month = np.arange(len(load)) // HPM

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[1, 1:])
    ax4 = fig.add_subplot(gs[2, :2]); ax5 = fig.add_subplot(gs[2, 2])

    # (1) Timeline: monthly average MW by source, demand, grid limit, milestones
    mm = np.arange(120)
    mavg = lambda x: np.bincount(month, weights=x, minlength=120) / HPM
    ax1.stackplot(mm, [mavg(s[k]) for k in used], colors=[COLORS[k] for k in used], labels=[LABELS[k] for k in used],
                  step='post', alpha=0.9)
    ax1.step(mm, mavg(load), where='post', color='k', lw=2, label='Demand (monthly avg)')
    ax1.step(mm, _monthly_max(load), where='post', color='k', lw=1, ls=':', label='Demand (monthly peak)')
    ax1.step(mm, _monthly_max(ordinary), where='post', color='#444', lw=1.5, ls='--', label='Grid import limit')
    caps = r['capacities']
    events = {start: f'Opening {md(start)}', start + 10: f'Full demand {md(start + 10)}'}
    for rid in RES:
        if caps.get(rid, 0) > 0:
            m0 = max(start, int(inp.opt.loc[rid, 'available_month']))
            events.setdefault(m0, ''); events[m0] = (events[m0] + ' / ' if events[m0] else '') + f'{LABELS[rid]} {md(m0)}'
    ymax = max(mavg(load).max(), ordinary.max()) * 1.35
    for i, (m0, txt) in enumerate(sorted(events.items())):
        ax1.axvline(m0, color='#666', lw=0.8, alpha=0.6)
        ax1.text(m0 + 0.5, ymax * (0.97 - 0.07 * (i % 4)), txt, fontsize=8, va='top')
    ax1.set_xlim(0, 120); ax1.set_ylim(0, ymax)
    if cal:
        ticks = list(range(0, 121, 12)); ax1.set_xticks(ticks); ax1.set_xticklabels([f'm{t_}\n{md(t_)}' for t_ in ticks], fontsize=8)
    ax1.set_xlabel('Development month (730-hour model months)' + (f' | month 0 = {md(0)}' if cal else '')); ax1.set_ylabel('MW (monthly average)')
    ax1.set_title('1. Timeline: how the portfolio serves the commissioning ramp over 10 years')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    for y in range(1, 10):
        ax1.axvline(12 * y, color='#ddd', lw=0.5, zorder=0)

    # (2) Portfolio capacities
    rc = list(RES)                                         # show every resource, selected or not
    cby = r.get('capacities_by_year') or [caps] * 10
    bars = ax2.barh([BAR_LABELS[k] for k in rc], [caps[k] for k in rc], color=[COLORS[k] for k in rc])
    xmax = max(max(caps.values()), 1)
    for b, k in zip(bars, rc):
        if caps[k] <= 0:
            txt = 'not selected'
        else:
            steps, prev = [], 0
            for y, cy in enumerate(cby):
                if cy.get(k, 0) > prev + 1e-9:
                    steps.append(f"Y{y + 1} +{cy[k] - prev:g}"); prev = cy[k]
            when = md(max(start, int(inp.opt.loc[k, 'available_month'])))
            txt = f"{caps[k]:g} MW from {when}" + (f"  ({', '.join(steps)})" if r.get('staged') and len(steps) > 1 else '')
        ax2.text(max(b.get_width(), 0) + xmax * 0.02, b.get_y() + b.get_height() / 2, txt, va='center', fontsize=8,
                 color='#888' if caps[k] <= 0 else 'k')
    ax2.set_xlim(0, xmax * 2.2); ax2.invert_yaxis()
    ax2.set_xlabel('MW'); ax2.set_title(('2. Final portfolio, staged by year' if r.get('staged') else '2. Optimised portfolio') + f'\n(land {r.get("land_acres", 0):.1f} of {inp.land:g} acres)')

    # (3) Typical mature week (year 8, mid-April)
    h0 = 7 * 8760 + 2400; w = slice(h0, h0 + 168); x = np.arange(168)
    ax3.stackplot(x, [s[k][w] for k in used], colors=[COLORS[k] for k in used], alpha=0.9)
    ax3.plot(x, load[w], 'k', lw=1.8, label='Demand')
    ax3.plot(x, eff[w], color='#444', ls='--', lw=1.2, label='Grid import limit')
    ax3.set_xlim(0, 167); ax3.set_xticks(range(0, 169, 24)); ax3.set_xticklabels([f'Day {i + 1}' for i in range(8)])
    ax3.set_ylabel('MW'); ax3.set_title(f'3. Typical week at full demand (model year 8, hours {h0}-{h0 + 167})')
    ax3.legend(loc='upper right', fontsize=8)

    # (4) Stress events: model year 8 window covering the August-September events
    ys = 7 * 8760; act = np.nonzero(active[ys:ys + 8760])[0]
    # window: 16 days from the day before the event start whose window holds the most stress hours (earliest on ties;
    # for the packet that is the first event); no stress events: the same mid-August window
    starts = act[np.r_[True, np.diff(act) > 1]] if act.size else np.array([5544])
    first = max(starts, key=lambda s0: (((act >= s0 - 24) & (act < s0 - 24 + 24 * 16)).sum(), -s0))
    a = max(ys, ys + first - 24); b = min(ys + 8760, a + 24 * 16)
    w = slice(a, b); x = np.arange(b - a)
    ax4.stackplot(x, [s[k][w] for k in used], colors=[COLORS[k] for k in used], alpha=0.9)
    ax4.plot(x, load[w], 'k', lw=1.8, label='Requested demand')
    ax4.plot(x, eff[w], color='#444', ls='--', lw=1.2, label='Effective grid limit (reduced in events)')
    ev = active[w].astype(float)
    ax4.fill_between(x, 0, 1, where=ev > 0, color='#f26b1d', alpha=0.18, step='mid', label='Stress event',
                     transform=ax4.get_xaxis_transform())
    ax4.set_xlim(0, b - a - 1); ax4.set_xticks(range(0, b - a + 1, 24)); ax4.set_xticklabels([f'D{i + 1}' for i in range((b - a) // 24 + 1)])
    ax4.set_ylabel('MW'); ax4.set_title('4. Grid-stress period (model year 8): import capped, onsite and flexibility respond' if act.size
                                        else '4. No stress events in this run (mid-August, model year 8)')
    ax4.legend(loc='upper right', fontsize=8)

    # (5) Annual cost and CO2
    ann = r['annual']; yrs = [a_['model_year'] for a_ in ann]
    cost = [a_['total_decision_cost_usd'] / max(a_['requested_mwh'], 1e-9) if a_['requested_mwh'] > 0 else np.nan for a_ in ann]
    co2 = [a_['combined_electricity_co2_kg_per_mwh'] if a_['combined_electricity_co2_kg_per_mwh'] is not None else np.nan for a_ in ann]
    ax5.bar(yrs, cost, color='#9a9a96', alpha=0.8)
    ax5.set_xlabel('Model year'); ax5.set_ylabel('\\$ / requested MWh', color='#555')
    ax5b = ax5.twinx(); ax5b.plot(yrs, co2, 'o-', color='#2e7d4f'); ax5b.set_ylabel('kg CO2 / MWh', color='#2e7d4f')
    if r.get('co2_cap_kg_mwh') is not None:
        ax5b.axhline(r['co2_cap_kg_mwh'], color='#2e7d4f', ls=':', lw=1)
        ax5b.text(yrs[0], r['co2_cap_kg_mwh'], f" ceiling {r['co2_cap_kg_mwh']:g} (10-yr)", color='#2e7d4f', fontsize=8, va='bottom')
    else:
        ax5b.axhline(100, color='#2e7d4f', ls=':', lw=1)
    ax5.set_title('5. Annual cost and emissions intensity')

    head = title or f"{r.get('reliability', '')}"
    fig.suptitle(f"Powerllel optimised portfolio  |  {inp.peak:g} MW campus  |  {head}\n"
                 f"Opening {md(start)}, full demand {md(start + 10)}  |  10-yr decision cost \\${r['total_decision_cost_usd'] / 1e6:,.1f}M "
                 f"(\\${r['decision_cost_usd_per_requested_mwh']:.1f}/MWh)  |  {r['combined_electricity_co2_kg_per_mwh']:.0f} kg CO2/MWh  |  "
                 f"clean {r['clean_energy_share']:.0%}  |  interrupted {r['interrupted_mwh']:,.0f} MWh  |  checker {r['checker']}",
                 fontsize=12)
    if save:
        os.makedirs('outputs', exist_ok=True)
        fn = f"outputs/dashboard_m{start}_{(title or 'run').replace(' ', '_').replace('/', '-')}.png"
        fig.savefig(fn, dpi=130); print('saved', os.path.abspath(fn))
    if show:
        plt.show()
    return fig


SRC_COLOR = {**{BAR_LABELS[k]: COLORS[k] for k in RES}, 'Merchant grid': COLORS['grid'],
             'Tier 1 relocation': COLORS['tier1_relocated'], 'Tier 1 relocated (remote)': COLORS['tier1_relocated'],
             'Tier 2 interruption': COLORS['tier2_curtailed']}
PART_COLOR = {'annualised_capex_usd': '#3D5A80', 'fixed_om_usd': '#98C1D9', 'energy_payment_usd': '#F2CC8F',
              'fuel_usd': '#E07A5F', 'grid_energy_usd': '#8C8C8C', 'demand_response_usd': '#5DADE2'}


def _savefig(fig, kind, r, title):
    os.makedirs('outputs', exist_ok=True)
    fn = f"outputs/{kind}_m{r['initial_service_month']}_{(title or 'run').replace(' ', '_').replace('/', '-')}.png"
    fig.savefig(fn, dpi=130); print('saved', os.path.abspath(fn))


def plot_capex(r, save=True, show=True, title=None):
    """Left: capex RATE of each source ($/kW, catalogue after overrides; offsite PPAs show their
    contract price instead). Right: upfront capex of the capacity built ($M = MW x $/kW x 1000)."""
    import matplotlib.pyplot as plt
    df = r['_cost_df']; df = df[df.option_id.isin(RES)].reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    cols = [SRC_COLOR[s_] for s_ in df.source]

    rate = df.capex_usd_per_kw.to_numpy(float); top1 = max(rate.max(), 1e-6)
    bars = a1.barh(df.source, rate, color=cols, alpha=0.9)
    for b, rt, x in zip(bars, rate, df.itertuples()):
        if rt > 0:
            txt = f'\\${rt:,.0f}/kW' + (f'  + \\${x.fom_usd_per_kw_year:,.0f}/kW-yr O&M' if x.fom_usd_per_kw_year > 0 else '')
        elif x.location == 'offsite':
            txt = f'PPA contract \\${x.payment_usd_per_mwh:,.0f}/MWh (no capex)'
        else:
            txt = 'no capex'
        txt += '' if x.mw > 0 else '   [not selected]'
        a1.text(b.get_width() + top1 * 0.02, b.get_y() + b.get_height() / 2, txt, va='center', fontsize=9,
                color='k' if x.mw > 0 else '#888')
    a1.set_xlim(0, top1 * 2.0); a1.invert_yaxis(); a1.set_xlabel('Capex rate, $ per kW')
    a1.set_title('Capex rate by source ($/kW)')

    v = df.upfront_capex_usd.to_numpy() / 1e6; top = max(v.max(), 1e-6)
    bars = a2.barh(df.source, v, color=cols)
    for b, val, x in zip(bars, v, df.itertuples()):
        txt = (f'\\${val:,.1f}M  ({x.mw:g} MW x \\${x.capex_usd_per_kw:,.0f}/kW)' if val > 0 else
               ('contract - no upfront capex' if x.mw > 0 else 'not selected'))
        a2.text(b.get_width() + top * 0.02, b.get_y() + b.get_height() / 2, txt, va='center', fontsize=9,
                color='k' if x.mw > 0 else '#888')
    a2.set_xlim(0, top * 1.9); a2.invert_yaxis(); a2.set_xlabel('Upfront capex, $ million')
    a2.set_title(f'Upfront capex by source  |  total \\${v.sum():,.1f}M')
    fig.suptitle(f"Capital cost  |  {title or r.get('reliability', '')}")
    if save: _savefig(fig, 'capex', r, title)
    if show: plt.show()
    return fig


def plot_carbon_intensity(r, save=True, show=True, title=None):
    """Carbon intensity (kg CO2 per MWh delivered) of every source. Sources that ran use the
    realised value from dispatch; sources that did not run show their catalogue factor (hatched).
    The battery has no direct emissions (its charging energy is counted in the supplying source)."""
    import matplotlib.pyplot as plt
    inp = r['_inp']; df = r['_co2_df'].set_index('source')
    rows = []
    for k in RES:
        lab = BAR_LABELS[k]
        if k == 'battery_4h':
            rows.append((lab, 0.0, df.loc[lab, 'mwh'] > 0 if lab in df.index else False, 'no direct emissions'))
            continue
        used = lab in df.index and df.loc[lab, 'mwh'] > 1e-6
        val = float(df.loc[lab, 'kg_co2_per_mwh']) if used else 1000 * _f(inp.opt.loc[k, 'co2_t_per_mwh'])
        rows.append((lab, val, used, ''))
    g_used = 'Merchant grid' in df.index and df.loc['Merchant grid', 'mwh'] > 1e-6
    g_val = float(df.loc['Merchant grid', 'kg_co2_per_mwh']) if g_used else 1000 * float(inp.grid_co2.mean())
    rows.append(('Merchant grid', g_val, g_used, 'hourly grid factor, energy-weighted' if g_used else 'average grid factor'))
    t1 = 'Tier 1 relocated (remote)'
    t_used = t1 in df.index and df.loc[t1, 'mwh'] > 1e-6
    rows.append((t1, float(df.loc[t1, 'kg_co2_per_mwh']) if t_used else 1000 * float(inp.remote_co2.mean()), t_used, 'remote site'))
    rows.sort(key=lambda z: -z[1])
    labs = [z[0] for z in rows]; vals = np.array([z[1] for z in rows]); top = max(vals.max(), 1e-6)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    bars = ax.barh(labs, vals, color=[SRC_COLOR[l_] for l_ in labs])
    for b, (lab, val, used, note) in zip(bars, rows):
        if not used:
            b.set_alpha(0.4); b.set_hatch('//')
        txt = f'{val:,.0f} kg/MWh' + (f'  ({note})' if note else '') + ('' if used else '  [not used - catalogue value]')
        ax.text(b.get_width() + top * 0.02, b.get_y() + b.get_height() / 2, txt, va='center', fontsize=9,
                color='k' if used else '#888')
    port = r['combined_electricity_co2_kg_per_mwh']
    ax.axvline(port, color='k', ls='--', lw=1.2)
    ax.text(port, len(labs) - 0.4, f' portfolio {port:.0f} kg/MWh', fontsize=9, va='bottom')
    ax.set_xlim(0, top * 1.8); ax.invert_yaxis(); ax.set_xlabel('Carbon intensity, kg CO2 per MWh delivered (operational)')
    ax.set_title(f"Carbon intensity by source  |  {title or r.get('reliability', '')}")
    if save: _savefig(fig, 'carbon_intensity', r, title)
    if show: plt.show()
    return fig


def plot_lifecycle(r, save=True, show=True, title=None):
    """10-year life-cycle cost by source, stacked by cost component."""
    import matplotlib.pyplot as plt
    df = r['_cost_df']; df = df[df.lifecycle_usd.abs() > 1].sort_values('lifecycle_usd', ascending=False)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(df) + 2.2), constrained_layout=True)
    left = np.zeros(len(df))
    for p_ in COST_PARTS:
        v = df[p_].to_numpy() / 1e6
        if np.abs(v).sum() > 1e-9:
            ax.barh(df.source, v, left=left, color=PART_COLOR[p_], label=COST_PART_LABELS[p_]); left += v
    top = max(left.max(), 1e-6)
    for y_, (tot, pm) in enumerate(zip(left, df.lifecycle_usd_per_mwh)):
        ax.text(tot + top * 0.01, y_, f'\\${tot:,.1f}M' + (f'  (\\${pm:,.0f}/MWh)' if np.isfinite(pm) else ''), va='center', fontsize=9)
    ax.set_xlim(0, top * 1.35); ax.invert_yaxis(); ax.set_xlabel('10-year life-cycle cost, $ million')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(f"10-year life-cycle cost by source  |  total \\${left.sum():,.1f}M "
                 f"(\\${r['decision_cost_usd_per_requested_mwh']:.1f}/requested MWh)  |  {title or r.get('reliability', '')}")
    if save: _savefig(fig, 'lifecycle', r, title)
    if show: plt.show()
    return fig


def plot_emissions(r, save=True, show=True, title=None):
    """CO2 by source: horizon total and by model year."""
    import matplotlib.pyplot as plt
    df = r['_co2_df']; ann = r['_co2_annual']
    keep = [s_ for s_ in df.source if df.set_index('source').co2_t[s_] > 1e-6]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True, gridspec_kw={'width_ratios': [1, 1.4]})
    d_ = df.set_index('source').loc[keep] if keep else df.set_index('source').iloc[:0]
    v = d_.co2_t.to_numpy() / 1e3; top = max(v.max() if v.size else 0, 1e-6)
    bars = a1.barh(d_.index, v, color=[SRC_COLOR[s_] for s_ in d_.index])
    for b, val, sh, kg in zip(bars, v, d_.share, d_.kg_co2_per_mwh):
        a1.text(b.get_width() + top * 0.02, b.get_y() + b.get_height() / 2, f'{val:,.1f} kt  ({sh:.0%}, {kg:.0f} kg/MWh)', va='center', fontsize=9)
    a1.set_xlim(0, top * 1.7); a1.invert_yaxis(); a1.set_xlabel('kt CO2 over the horizon')
    a1.set_title(f'CO2 by source  |  total {v.sum():,.1f} kt')
    bottom = np.zeros(len(ann))
    for s_ in keep:
        a2.bar(ann.index, ann[s_] / 1e3, bottom=bottom, color=SRC_COLOR[s_], label=s_); bottom += ann[s_].to_numpy() / 1e3
    a2.set_xlabel('Model year'); a2.set_ylabel('kt CO2'); a2.set_xticks(list(ann.index))
    a2.set_title('CO2 by source and model year'); a2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Operational emissions  |  {r['combined_electricity_co2_kg_per_mwh']:.0f} kg CO2/MWh  |  {title or r.get('reliability', '')}")
    if save: _savefig(fig, 'co2', r, title)
    if show: plt.show()
    return fig


def print_cost_and_co2(r):
    cdf = r['_cost_df']; edf = r['_co2_df']
    M = lambda x: f'{x / 1e6:,.1f}'
    print('\n   UPFRONT CAPEX AND 10-YEAR LIFE-CYCLE COST BY SOURCE ($ million)')
    hdr = (f"   {'Source':<21}{'MW':>6}{'Capex':>9} |{'Ann.capex':>10}{'Fix O&M':>9}{'Energy':>9}{'Fuel':>8}"
           f"{'Grid':>8}{'DR':>7}{'LIFE-CYCLE':>12}{'$/MWh':>7}")
    print(hdr); print('   ' + '-' * (len(hdr) - 3))
    sh = cdf[(cdf.lifecycle_usd.abs() > 1) | (cdf.upfront_capex_usd > 0) | (cdf.mw > 0)]
    for _, x in sh.iterrows():
        pm = f'{x.lifecycle_usd_per_mwh:7.0f}' if np.isfinite(x.lifecycle_usd_per_mwh) else '      -'
        print(f"   {x.source:<21}{x.mw:6g}{M(x.upfront_capex_usd):>9} |{M(x.annualised_capex_usd):>10}{M(x.fixed_om_usd):>9}"
              f"{M(x.energy_payment_usd):>9}{M(x.fuel_usd):>8}{M(x.grid_energy_usd):>8}{M(x.demand_response_usd):>7}"
              f"{M(x.lifecycle_usd):>12}{pm}")
    print('   ' + '-' * (len(hdr) - 3))
    tot = cdf.sum(numeric_only=True)
    print(f"   {'TOTAL':<21}{'':6}{M(tot.upfront_capex_usd):>9} |{M(tot.annualised_capex_usd):>10}{M(tot.fixed_om_usd):>9}"
          f"{M(tot.energy_payment_usd):>9}{M(tot.fuel_usd):>8}{M(tot.grid_energy_usd):>8}{M(tot.demand_response_usd):>7}"
          f"{M(tot.lifecycle_usd):>12}")
    gap = tot.lifecycle_usd - r.get('total_decision_cost_usd', tot.lifecycle_usd)
    print(f'   (life-cycle total reconciles to the 10-yr decision cost within ${gap:,.0f})')
    print('\n   CO2 EMISSIONS BY SOURCE (operational)')
    print(f"   {'Source':<27}{'GWh':>9}{'kt CO2':>10}{'share':>8}{'kg/MWh':>9}")
    for _, x in edf.iterrows():
        if x.mwh > 0 or x.co2_t > 0:
            print(f"   {x.source:<27}{x.mwh / 1e3:9,.1f}{x.co2_t / 1e3:10,.2f}{x.share:8.0%}{x.kg_co2_per_mwh:9.0f}")
    print(f"   {'TOTAL':<27}{'':9}{edf.co2_t.sum() / 1e3:10,.2f}")


def plot_years(r, d, save=True, show=True, inp=None, title=None):
    """One panel per model year (stacked vertically): daily-average supply mix vs demand and
    grid limit, stress days shaded, with that year's capacities, cost, CO2 and clean share."""
    import matplotlib.pyplot as plt
    start = r['initial_service_month']
    inp = inp or r.get('_inp') or Inputs(Scenario(start=start, grid_delay=r.get('grid_delay_months', 0)))
    cal = inp.cu.calendar_start; md = lambda m_: month_to_date(m_, cal)
    s = _supply(d); used = [k for k in STACK if s[k].sum() > 1e-6]
    years = [y for y in range(10) if inp.load[y * 8760:(y + 1) * 8760].sum() > 0]
    ann = {a_['model_year']: a_ for a_ in r['annual']}
    cby = r.get('capacities_by_year') or [r['capacities']] * 10
    ymax = max(inp.load.max(), inp.cap_ord.max()) * 1.12
    fig, axes = plt.subplots(len(years), 1, figsize=(16, 2.9 * len(years)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    day = lambda x: x.reshape(365, 24).mean(axis=1)
    xd = np.arange(365)
    for ax, y in zip(axes, years):
        sl = slice(y * 8760, (y + 1) * 8760)
        ax.stackplot(xd, [day(s[k][sl]) for k in used], colors=[COLORS[k] for k in used],
                     labels=[LABELS[k] for k in used], alpha=0.9)
        ax.plot(xd, day(inp.load[sl]), 'k', lw=1.4, label='Demand (daily avg)')
        ax.plot(xd, inp.load[sl].reshape(365, 24).max(axis=1), 'k:', lw=0.8, label='Demand (daily peak)')
        ax.plot(xd, inp.cap_ord[sl].reshape(365, 24).max(axis=1), color='#444', ls='--', lw=1.1, label='Grid import limit')
        ev = inp.active[sl].reshape(365, 24).any(axis=1)
        ax.fill_between(xd, 0, 1, where=ev, color='#f26b1d', alpha=0.25, step='mid', transform=ax.get_xaxis_transform(),
                        label='Stress-event day')
        a_ = ann.get(y + 1, {})
        cap_txt = ', '.join(f'{BAR_LABELS[k]} {v:g}' for k, v in cby[y].items() if v > 0 and inp.mask[k][sl].any()) or 'no capacity in service'
        m0, m1 = 12 * y, 12 * y + 11
        cost = a_.get('total_decision_cost_usd', 0) / a_['requested_mwh'] if a_.get('requested_mwh') else float('nan')
        co2 = a_.get('combined_electricity_co2_kg_per_mwh') or float('nan')
        cl = a_.get('clean_energy_share') or 0
        ax.set_title(f'Model year {y + 1}  ({md(m0)} to {md(m1)})   |   \\${cost:.1f}/MWh   {co2:.0f} kg CO2/MWh   '
                     f'clean {cl:.0%}   |   in service: {cap_txt}', fontsize=10, loc='left')
        ax.set_ylim(0, ymax); ax.set_ylabel('MW')
        for mm in range(1, 12):
            ax.axvline(mm * 730 / 24, color='#ddd', lw=0.5, zorder=0)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    axes[-1].set_xlim(0, 364)
    axes[-1].set_xticks([mm * 730 / 24 for mm in range(12)])
    axes[-1].set_xticklabels([f'month {mm + 1}' for mm in range(12)])
    axes[-1].set_xlabel('Month within model year (daily averages)')
    fig.suptitle(f"Year-by-year operation  |  {inp.peak:g} MW campus  |  {title or r.get('reliability', '')}"
                 + ('  |  capacity staged year by year' if r.get('staged') else ''), fontsize=13)
    if save:
        os.makedirs('outputs', exist_ok=True)
        fn = f"outputs/years_m{start}_{(title or 'run').replace(' ', '_').replace('/', '-')}.png"
        fig.savefig(fn, dpi=110); print('saved', os.path.abspath(fn))
    if show:
        plt.show()
    return fig
