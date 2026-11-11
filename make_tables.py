#!/usr/bin/env python3
"""
make_tables.py — sinh các bảng LaTeX từ file CSV do notebook xuất ra.

CÁCH DÙNG
---------
1. Chạy notebook mô hình hoá trên Kaggle (MODE = 'baseline').
2. Tải thư mục Output về, đặt các file CSV vào ./models/ cạnh script này:
       models/cv_results.csv
       models/tuning_log.csv
       models/test_results.csv
3. Chạy:  python make_tables.py
4. Script ghi các file tabular vào ./tables/ và in ra khối macro
   để dán đè vào phần "ALL NUMERIC RESULTS LIVE HERE" trong main.tex.

Script chỉ sinh phần `tabular`; phần `\\begin{table}`, caption và label đã
nằm sẵn trong main.tex.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

CSV_DIR = 'models'
OUT_DIR = 'tables'

# Nhãn cột hiển thị trong bảng LaTeX. Hậu tố cho biết thang đo.
NHAN_CV = {
    'mo_hinh':             'Model',
    'rmsle_mean':          r'RMSLE (log) $\downarrow$',
    'rmsle_std':           r'$\pm$ s.d.',
    'rmsle_train':         'Train (log)',
    'mae_seconds_mean':    r'MAE (s) $\downarrow$',
    'rmse_seconds_mean':   r'RMSE (s) $\downarrow$',
    'r2_mean':             r'$R^2$ (log) $\uparrow$',
    'giay':                'Time (s)',
}
NHAN_TUNING = {
    'mo_hinh':          'Model',
    'rmsle_cv_truoc':   'Before',
    'rmsle_cv_sau':     'After',
    'cai_thien_%':      r'Change in RMSLE (\%)',
    'cau_hinh_sau':     'Selected configuration',
}
# Lưu ý: cột 'vi_pham' trong audit.csv trùng nội dung với 'domain',
# nên cố ý không liệt kê ở đây để bảng khỏi thừa một cột.
NHAN_AUDIT = {
    'cot':          'Column',
    'kieu':         'Type',
    'thieu':        'Missing',
    'thieu_an':     'Disguised missing',
    'iqr':          'Flagged by IQR',
    'zscore':       r'Flagged by $|z|>3$',
    'domain':       'Removed by domain rule',
}
NHAN_BASELINE = {
    'mo_hinh':   'Model',
    'cau_hinh':  'Non-default settings',
}
NHAN_HOLDOUT = {
    'mo_hinh':               'Model',
    'Model':                 'Model',
    'Holdout_RMSLE':         'RMSLE',
    'Holdout_MAE_seconds':   'MAE (s)',
    'Holdout_RMSE_seconds':  'RMSE (s)',
    'Holdout_R2':            r'$R^2$',
    'Training_time':         'Fit (s)',
}

# Số chữ số thập phân theo từng cột
# Dịch tên mô hình sang tiếng Anh (báo cáo viết bằng tiếng Anh)
DICH_TEN = {
    'XGBoost – train toàn bộ dev': 'XGBoost (full development set)',
    'XGBoost - train toan bo dev': 'XGBoost (full development set)',
    'Dummy (mean)': 'Mean predictor',
}

LAM_TRON = {
    'rmsle_mean': 4, 'rmsle_std': 4, 'rmsle_train': 4, 'r2_mean': 4,
    'mae_seconds_mean': 1, 'rmse_seconds_mean': 1, 'giay': 1,
    'rmsle_cv_truoc': 4, 'rmsle_cv_sau': 4, 'cai_thien_%': 2,
    'Holdout_RMSLE': 4, 'Holdout_R2': 4,
    'Holdout_MAE_seconds': 1, 'Holdout_RMSE_seconds': 1, 'Training_time': 1,
    # Bảng audit: các cột đếm, in dạng số nguyên có dấu phân cách nghìn
    'thieu': 0, 'thieu_an': 0, 'vi_pham': 0,
    'iqr': 0, 'zscore': 0, 'domain': 0,
}


def dinh_dang_so(v, n: int) -> str:
    """Định dạng số với n chữ số thập phân.

    Loại bỏ 'số không âm': -0.0000 sau khi làm tròn phải in là 0.0000, vì
    dấu trừ ở đây là nhiễu của dấu phẩy động chứ không mang ý nghĩa.
    """
    if pd.isna(v):
        return '--'
    s = f'{v:,.{n}f}'
    if set(s) <= set('-0,.') and s.startswith('-'):
        s = s[1:]
    return s


def thoat_latex(s) -> str:
    """Thoát các ký tự đặc biệt của LaTeX trong chuỗi."""
    s = str(s)
    for goc, thay in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                      ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'),
                      ('}', r'\}'), ('~', r'\textasciitilde{}'),
                      ('^', r'\textasciicircum{}')]:
        s = s.replace(goc, thay)
    return s


def lam_gon_cau_hinh(s) -> str:
    """Rút gọn chuỗi dict tham số thành dạng `key=value` ngắn gọn, dễ xuống dòng."""
    import re as _re
    s = str(s)
    # Bỏ lớp bọc np.float64(...) / np.int64(...) do NumPy sinh ra
    s = _re.sub(r'np\.\w+\(([^()]*)\)', r'\1', s)
    s = s.strip().strip('{}')
    phan = []
    for cap in s.split(','):
        if ':' not in cap:
            continue
        k, v = cap.split(':', 1)
        phan.append(f"{k.strip().strip(chr(39)).strip('\"')}="
                    f"{v.strip().strip(chr(39)).strip(chr(34))}")
    return ', '.join(phan) if phan else str(s)


def doc_csv(ten_file: str) -> pd.DataFrame | None:
    duong_dan = os.path.join(CSV_DIR, ten_file)
    if not os.path.exists(duong_dan):
        print(f'  [bỏ qua] không tìm thấy {duong_dan}')
        return None
    return pd.read_csv(duong_dan)


def sinh_tabular(df: pd.DataFrame, nhan: dict, in_dam_dong: int | None = None,
                 cot_trai: int = 1, canh_cot: str | None = None) -> str:
    """Chuyển DataFrame thành phần tabular dùng booktabs.

    canh_cot: chuỗi căn cột LaTeX tự đặt (VD 'lrrr p{5cm}'); để None thì cột
    đầu căn trái, các cột còn lại căn phải.
    """
    cot = [c for c in nhan if c in df.columns]
    d = df[cot].copy()

    for c in cot:
        if c in LAM_TRON and pd.api.types.is_numeric_dtype(d[c]):
            n = LAM_TRON[c]
            d[c] = d[c].map(lambda v: dinh_dang_so(v, n))
        else:
            d[c] = d[c].map(thoat_latex)

    canh = canh_cot or ('l' * cot_trai + 'r' * (len(cot) - cot_trai))
    dong = ['\\begin{tabular}{@{}' + canh + '@{}}', '\\toprule',
            ' & '.join(nhan[c] for c in cot) + r' \\', '\\midrule']

    for i, (_, r) in enumerate(d.iterrows()):
        o = [str(v) for v in r.tolist()]
        if in_dam_dong is not None and i == in_dam_dong:
            o = [f'\\textbf{{{v}}}' for v in o]
        dong.append(' & '.join(o) + r' \\')

    dong += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(dong) + '\n'


def ghi(ten: str, noi_dung: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    duong_dan = os.path.join(OUT_DIR, ten)
    with open(duong_dan, 'w', encoding='utf-8') as f:
        f.write(noi_dung)
    print(f'  [ghi]  {duong_dan}')


def main() -> int:
    print('Đang sinh bảng LaTeX...\n')
    macro: dict[str, str] = {}

    # ---------- Bảng 1: cross-validation ----------
    cv = doc_csv('cv_results.csv')
    if cv is not None:
        cv['mo_hinh'] = cv['mo_hinh'].map(lambda v: DICH_TEN.get(str(v).strip(), v))
        cv = cv.sort_values('rmsle_mean').reset_index(drop=True)
        ghi('table_cv.tex', sinh_tabular(cv, NHAN_CV, in_dam_dong=0))

        tot = cv.iloc[0]
        macro['BestModel'] = thoat_latex(tot['mo_hinh'])
        macro['BestCVRMSLE'] = f"{tot['rmsle_mean']:.4f}"
        macro['BestCVStd'] = f"{tot['rmsle_std']:.4f}"

        for khoa, ten in [('LinearCVRMSLE', 'Linear (baseline)'),
                          ('DummyCVRMSLE', 'Mean predictor')]:
            hang = cv[cv['mo_hinh'] == ten]
            if len(hang):
                macro[khoa] = f"{hang.iloc[0]['rmsle_mean']:.4f}"
                goc = hang.iloc[0]['rmsle_mean']
                ten_gain = 'GainOverLinear' if 'Linear' in ten else 'GainOverDummy'
                macro[ten_gain] = f"{(goc - tot['rmsle_mean']) / goc * 100:.1f}\\%"

    # ---------- Bảng 2: tuning log ----------
    tn = doc_csv('tuning_log.csv')
    if tn is not None:
        # CSV ghi 'cai_thien_%' am khi ket qua xau di. Bai bao dung quy uoc
        # nguoc lai cho de doc: duong = RMSLE tang = te hon.
        if 'cai_thien_%' in tn.columns:
            tn = tn.copy()
            tn['cai_thien_%'] = -tn['cai_thien_%']
        if 'cau_hinh_sau' in tn.columns:
            tn = tn.copy()
            tn['cau_hinh_sau'] = tn['cau_hinh_sau'].map(lam_gon_cau_hinh)
        # Cột cấu hình dùng p{} để tự xuống dòng, tránh tràn lề trang
        ghi('table_tuning.tex',
            sinh_tabular(tn, NHAN_TUNING, canh_cot='lrrr>{\\raggedright\\arraybackslash}p{7.2cm}'))

    # ---------- Bảng 3: holdout ----------
    ho = doc_csv('test_results.csv')
    if ho is not None:
        cot_ten = 'mo_hinh' if 'mo_hinh' in ho.columns else 'Model'
        ho[cot_ten] = ho[cot_ten].map(lambda v: DICH_TEN.get(str(v).strip(), v))
        ho = ho.sort_values('Holdout_RMSLE').reset_index(drop=True)
        # In đậm MÔ HÌNH CUỐI: ưu tiên bản train trên toàn bộ development set
        i_dam = 0
        uu_tien = ho.index[ho[cot_ten].str.contains('full development set', na=False)]
        if len(uu_tien):
            i_dam = int(uu_tien[0])
        elif 'BestModel' in macro:
            khop = ho.index[ho[cot_ten].map(thoat_latex) == macro['BestModel']]
            if len(khop):
                i_dam = int(khop[0])
        ghi('table_holdout.tex', sinh_tabular(ho, NHAN_HOLDOUT, in_dam_dong=i_dam))

        r = ho.iloc[i_dam]
        macro['BestHoldRMSLE'] = f"{r['Holdout_RMSLE']:.4f}"
        macro['BestHoldMAE'] = f"{r['Holdout_MAE_seconds']:,.0f}"
        macro['BestHoldRMSE'] = f"{r['Holdout_RMSE_seconds']:,.0f}"
        macro['BestHoldRTwo'] = f"{r['Holdout_R2']:.4f}"

    # ---------- Bảng 4: audit chất lượng dữ liệu ----------
    au = doc_csv('audit.csv')
    if au is not None:
        ghi('table_audit.tex', sinh_tabular(au, NHAN_AUDIT, cot_trai=1))

    # ---------- Bảng 5: cấu hình mặc định của từng mô hình ----------
    bc = doc_csv('baseline_config.csv')
    if bc is not None:
        if 'cau_hinh' in bc.columns:
            bc = bc.copy()
            bc['cau_hinh'] = bc['cau_hinh'].map(lam_gon_cau_hinh)
        ghi('table_baselinecfg.tex',
            sinh_tabular(bc, NHAN_BASELINE,
                         canh_cot='l>{\\raggedright\\arraybackslash}p{5.6cm}'))

    # ---------- In khối macro để dán vào main.tex ----------
    if macro:
        print('\n' + '=' * 70)
        print('DÁN KHỐI SAU VÀO main.tex (thay các dòng \\newcommand tương ứng):')
        print('=' * 70)
        for k, v in macro.items():
            print(f'\\newcommand{{\\{k}}}{{{v}}}')
        print('=' * 70)
        print('Sau khi dán xong, đặt \\showtodofalse để tắt tô vàng khi nộp bài.')
    else:
        print('\nChưa sinh được macro nào — kiểm tra lại các file CSV trong ./models/')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
