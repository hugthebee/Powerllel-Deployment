"""Build the financial agent's method as an Excel workbook with live formulas (shown on the website's Methodology page).

    .venv/bin/python team_app/build_financial_workbook.py      # -> faraday_web/public/methodology/Powerllel_Financial_Model.xlsx

Sheets: Read me · Inputs (template, weights, limits) · Portfolios (the whole portfolio pool, each scored by formulas) · Recommendation.
The score is the Faraday pool method (team_app/logic.py rank): each goal is scaled 0-1 between its best and worst anchor, then weighted.
Change a weight or a limit in Inputs and Excel re-ranks every portfolio.
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / 'faraday_web' / 'public' / 'methodology' / 'Powerllel_Financial_Model.xlsx'

TEMPLATES = [('Balanced', 34, 33, 33), ('Speed-first', 70, 20, 10), ('Cost-first', 10, 80, 10), ('Carbon-first', 10, 20, 70)]
CAPS = [('utility_solar', 'Solar MW'), ('wind', 'Wind MW'), ('nuclear', 'Nuclear MW'), ('battery_4h', 'Battery MW'), ('gas_turbine', 'Gas turbine MW'), ('gas_engine', 'Gas engine MW'), ('fuel_cell', 'Fuel cell MW')]
HEAD = Font(bold=True, color='FFFFFF')
HEAD_FILL = PatternFill('solid', fgColor='111111')
INPUT_FILL = PatternFill('solid', fgColor='FDEBE0')
BOLD = Font(bold=True)


def main():
    with redirect_stdout(io.StringIO()):
        import faraday_service as S
        import logic as L
        pool = S._pool(False)
    an = L.anchors(pool)
    n = len(pool)
    d = L._axes(pool, pool.start.max(), 0.0, 0.0)

    wb = Workbook()
    # ---------------------------------------------------------------- Read me
    rm = wb.active
    rm.title = 'Read me'
    lines = [
        ('Powerllel financial agent: the Faraday pool method', True),
        ('', False),
        ('What it does', True),
        (f'1. A pool of {n:,} candidate portfolios for the 75 MW packet campus was built in advance (openings month 14-26). Each one was run through the', False),
        ('   hourly dispatch for all 87,600 hours, so its 10-year decision cost, emissions, clean share and merchant energy are known (sheet Portfolios).', False),
        ('2. Your limits remove portfolios: latest opening month, highest average emissions, most extra cost above the cheapest (sheet Inputs).', False),
        ('3. Each remaining portfolio is scored on three goals, each scaled 0 to 1 between its best and worst anchor:', False),
        ('      speed  = (worst full-demand month - its full-demand month) / (worst - best)', False),
        ('      cost   = (worst $/MWh - its $/MWh) / (worst - best)          $/MWh = 10-year decision cost / requested MWh', False),
        ('      carbon = (worst kg/MWh - its kg/MWh) / (worst - best)', False),
        ('   Each is clipped to 0-1. Score = 100 x (w_speed x speed + w_cost x cost + w_carbon x carbon) / (w_speed + w_cost + w_carbon).', False),
        ('4. The recommendation is the portfolio with the highest score (sheet Recommendation). Ties go to the cheaper one.', False),
        ('', False),
        ('Templates', True),
        ('The four customer types (Balanced, Speed-first, Cost-first, Carbon-first) come from the GridTech_Inputs.xlsx customer types. Pick one in Inputs!B4,', False),
        ('or type your own weights in Inputs!B10:B12 (leave them blank to use the template).', False),
        ('', False),
        ('Anchors (from the whole pool, not the survivors, so adding a limit never changes how the rest rank)', True),
        (f"   speed: best full demand month {an['speed'][0]:g}, worst {an['speed'][1]:g}", False),
        (f"   cost: best ${an['cost'][0]:.2f}/MWh, worst ${an['cost'][1]:.2f}/MWh (the cost of the cleanest portfolio at any opening)", False),
        (f"   carbon: best {an['carbon'][0]:.1f} kg/MWh, worst {an['carbon'][1]:.1f} kg/MWh (the emissions of the cheapest portfolio at any opening)", False),
        ('', False),
        ("Scope: the packet case on packet v6.0.1, the optimiser's packet (75 MW, ERCOT stress events, the packet grid schedule). Costs are 2023 USD, summed over 10 years without discounting.", False),
        ('The Powerllel website runs the same method live, plus price sensitivity (bear / base / bull) and the stress-event and backup checks.', False),
    ]
    for i, (t, b) in enumerate(lines, 1):
        rm.cell(i, 1, t).font = Font(bold=b, size=14 if i == 1 else 11)
    rm.column_dimensions['A'].width = 150

    # ---------------------------------------------------------------- Inputs
    ins = wb.create_sheet('Inputs')
    ins['A1'] = 'Inputs: change the orange cells'; ins['A1'].font = Font(bold=True, size=14)
    ins['A3'] = 'Template'; ins['A3'].font = BOLD
    ins['A4'] = 'Customer template'; ins['B4'] = 'Balanced'; ins['B4'].fill = INPUT_FILL
    dv = DataValidation(type='list', formula1='"' + ','.join(t[0] for t in TEMPLATES) + '"', allow_blank=False)
    ins.add_data_validation(dv); dv.add('B4')
    ins['D3'] = 'Template'; ins['E3'] = 'Speed'; ins['F3'] = 'Cost'; ins['G3'] = 'Carbon'
    for c in 'DEFG':
        ins[f'{c}3'].font = BOLD
    for i, (n, a, b, c) in enumerate(TEMPLATES, 4):
        ins[f'D{i}'], ins[f'E{i}'], ins[f'F{i}'], ins[f'G{i}'] = n, a, b, c
    ins['A9'] = 'Your own weights (optional, blank = template)'; ins['A9'].font = BOLD
    for r, lab, col in ((10, 'Speed weight', 'E'), (11, 'Cost weight', 'F'), (12, 'Carbon weight', 'G')):
        ins[f'A{r}'] = lab; ins[f'B{r}'].fill = INPUT_FILL
        ins[f'C{r}'] = f'=IF(B{r}="",VLOOKUP($B$4,$D$4:$G$7,{ "EFG".index(col) + 2 },FALSE),B{r})'
    ins['C9'] = 'Used'; ins['C9'].font = BOLD
    ins['A13'] = 'Weights total'; ins['C13'] = '=C10+C11+C12'
    ins['A15'] = 'Limits (blank = no limit)'; ins['A15'].font = BOLD
    ins['A16'] = 'Latest opening month (14-26)'; ins['B16'].fill = INPUT_FILL
    ins['A17'] = 'Highest average emissions, kg CO2/MWh'; ins['B17'].fill = INPUT_FILL
    ins['A18'] = 'Most extra cost above the cheapest that meets the other limits, %'; ins['B18'].fill = INPUT_FILL
    ins['A19'] = 'Cheapest portfolio meeting the opening and emissions limits, $M'
    ins['C19'] = '=MINIFS(Portfolios!D:D,Portfolios!T:T,1)'
    ins['A21'] = 'Anchors (fixed, from the whole pool)'; ins['A21'].font = BOLD
    ins['B21'] = 'best'; ins['C21'] = 'worst'
    for r, (lab, k) in zip((22, 23, 24), (('Speed: full-demand month', 'speed'), ('Cost: $ per requested MWh', 'cost'), ('Carbon: kg CO2/MWh', 'carbon'))):
        ins[f'A{r}'] = lab; ins[f'B{r}'] = round(an[k][0], 4); ins[f'C{r}'] = round(an[k][1], 4)
    ins.column_dimensions['A'].width = 62
    for c in 'BCDEFG':
        ins.column_dimensions[c].width = 14

    # ---------------------------------------------------------------- Portfolios
    ps = wb.create_sheet('Portfolios')
    cols = ['#', 'Opening month', 'Full-demand month', '10-yr decision cost $M', 'Requested GWh', '$ per requested MWh', 'kg CO2/MWh', 'Clean share %', 'Merchant share %',
            *[lab for _, lab in CAPS], 'Interrupted MWh', 'Relocated MWh',
            'Meets opening + emissions limits', 'Meets every limit', 'Speed score', 'Cost score', 'Carbon score', 'Score (0-100)', 'Rank']
    for j, c in enumerate(cols, 1):
        cell = ps.cell(1, j, c); cell.font = HEAD; cell.fill = HEAD_FILL; cell.alignment = Alignment(wrap_text=True, vertical='top')
    ps.freeze_panes = 'C2'
    n = len(d)
    rows = d.sort_values(['start', 'total_cost']).reset_index(drop=True)
    for i, r in rows.iterrows():
        x = i + 2
        vals = [i + 1, int(r.start), int(r.full_month), round(r.total_cost / 1e6, 3), round(r.requested_mwh / 1e3, 3), round(r.cost_axis, 4), round(r.intensity, 3),
                round(r.clean_share * 100, 2), round(r.merchant_mwh / r.served_mwh * 100, 2), *[float(r[k]) for k, _ in CAPS], round(r.interrupted_mwh, 1), round(r.relocated_mwh, 1)]
        for j, v in enumerate(vals, 1):
            ps.cell(x, j, v)
        # T: opening + emissions; U: every limit; V-X: goal scores; Y: score; Z: rank
        ps[f'T{x}'] = f'=IF(AND(OR(Inputs!$B$16="",B{x}<=Inputs!$B$16),OR(Inputs!$B$17="",G{x}<=Inputs!$B$17)),1,0)'
        ps[f'U{x}'] = f'=IF(AND(T{x}=1,OR(Inputs!$B$18="",D{x}<=Inputs!$C$19*(1+Inputs!$B$18/100))),1,0)'
        ps[f'V{x}'] = f'=MAX(0,MIN(1,(Inputs!$C$22-C{x})/(Inputs!$C$22-Inputs!$B$22)))'
        ps[f'W{x}'] = f'=MAX(0,MIN(1,(Inputs!$C$23-F{x})/(Inputs!$C$23-Inputs!$B$23)))'
        ps[f'X{x}'] = f'=MAX(0,MIN(1,(Inputs!$C$24-G{x})/(Inputs!$C$24-Inputs!$B$24)))'
        ps[f'Y{x}'] = f'=IF(U{x}=1,100*(Inputs!$C$10*V{x}+Inputs!$C$11*W{x}+Inputs!$C$12*X{x})/Inputs!$C$13-D{x}/1E9,-1)'   # tiny cost term: ties go to the cheaper
        ps[f'Z{x}'] = f'=IF(U{x}=1,RANK(Y{x},$Y$2:$Y${n + 1}),"")'
    for j in range(1, len(cols) + 1):
        ps.column_dimensions[get_column_letter(j)].width = 13
    ps.row_dimensions[1].height = 45

    # ---------------------------------------------------------------- Recommendation
    rc = wb.create_sheet('Recommendation')
    rc['A1'] = 'Recommended portfolio'; rc['A1'].font = Font(bold=True, size=14)
    rc['A2'] = 'The highest score in Portfolios among those meeting every limit. Change Inputs to re-rank.'
    rc['A4'] = 'Row in Portfolios'; rc['B4'] = f'=MATCH(MAX(Portfolios!Y2:Y{n + 1}),Portfolios!Y2:Y{n + 1},0)+1'
    items = [('Score', 'Y'), ('Opening month', 'B'), ('Full-demand month', 'C'), ('10-yr decision cost $M', 'D'), ('$ per requested MWh', 'F'), ('kg CO2/MWh', 'G'),
             ('Clean share %', 'H'), ('Merchant share %', 'I'), *[(lab, get_column_letter(10 + i)) for i, (_, lab) in enumerate(CAPS)]]
    for i, (lab, col) in enumerate(items, 5):
        rc[f'A{i}'] = lab
        rc[f'B{i}'] = f'=INDEX(Portfolios!{col}:{col},$B$4)'
    r0 = 5 + len(items) + 1
    rc[f'A{r0}'] = 'Best on each goal (among portfolios meeting every limit)'; rc[f'A{r0}'].font = BOLD
    rc[f'A{r0 + 1}'] = 'Cheapest, $M'; rc[f'B{r0 + 1}'] = f'=MINIFS(Portfolios!D:D,Portfolios!U:U,1)'
    rc[f'A{r0 + 2}'] = 'Earliest full demand, month'; rc[f'B{r0 + 2}'] = f'=MINIFS(Portfolios!C:C,Portfolios!U:U,1)'
    rc[f'A{r0 + 3}'] = 'Cleanest, kg CO2/MWh'; rc[f'B{r0 + 3}'] = f'=MINIFS(Portfolios!G:G,Portfolios!U:U,1)'
    rc[f'A{r0 + 4}'] = 'Portfolios meeting every limit'; rc[f'B{r0 + 4}'] = '=SUM(Portfolios!U:U)'
    rc.column_dimensions['A'].width = 52; rc.column_dimensions['B'].width = 18
    wb.move_sheet('Recommendation', offset=-2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)

    # check: the same formulas in pandas give the financial agent's own pick for each template
    for name, a, b, c in TEMPLATES:
        s = rows.copy()
        sc = lambda v, k: ((an[k][1] - v) / (an[k][1] - an[k][0])).clip(0, 1)
        s['score'] = 100 * (a * sc(s.full_month.astype(float), 'speed') + b * sc(s.cost_axis, 'cost') + c * sc(s.intensity, 'carbon')) / (a + b + c)
        mine = s.sort_values(['score', 'total_cost'], ascending=[False, True]).iloc[0]
        theirs = L.rank(pool, [a, b, c], ref=pool).iloc[0]
        same = all(abs(float(mine[k]) - float(theirs[k])) < 1e-9 for k, _ in CAPS) and mine.start == theirs.start
        print(f"{name:12s} workbook pick: opening {int(mine.start)}, ${mine.total_cost / 1e6:,.1f}M  |  agent pick: opening {int(theirs.start)}, ${theirs.total_cost / 1e6:,.1f}M  {'SAME' if same else 'DIFFERENT'}")
    print(f'wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {n:,} portfolios)')


if __name__ == '__main__':
    main()
