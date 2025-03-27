import pandas as pd

def run():
    file_path = file_path = r"D:\Listening skills\Bajaj Finserv\Data Engineering\Data Engineering\data.xlsx"
    df = pd.read_excel(file_path)
    df['attendance_date'] = pd.to_datetime(df['attendance_date'])
    df.sort_values(by=['student_id', 'attendance_date'], inplace=True)
    df_absent = df[df['status'].str.lower() == 'absent'].copy()
    df_absent['group'] = (
        df_absent.groupby('student_id')['attendance_date'].diff().dt.days.ne(1).cumsum()
    )

    streaks = (
        df_absent.groupby(['student_id','group'])
        .agg(
            absence_start_date=('attendance_date','min'),
            absence_end_date=('attendance_date','max'),
            total_absent_days=('attendance_date', 'count')
        ).reset_index()
    )
    filtered = streaks[streaks['total_absent_days'] > 3]
    result = (
        filtered.sort_values(['student_id', 'absence_end_date'], ascending=[True, False])
        .drop_duplicates('student_id')
        .reset_index(drop=True)
    )
    return result[['student_id', 'absence_start_date', 'absence_end_date', 'total_absent_days']]

output = run()
print(output)

