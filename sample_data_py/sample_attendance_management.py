import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample attendance management data
num_samples = 100
start_date = datetime.now() - timedelta(days=30)

# List of employees
employees = [f"EMP{i:03d}" for i in range(1, 21)]

# Generate attendance data
attendance_data = []

for date in [start_date + timedelta(days=i) for i in range(30)]:
    for emp in employees:
        # Determine attendance status
        is_present = np.random.choice([True, False], p=[0.9, 0.1])
        
        if is_present:
            # If present, generate check-in and check-out times
            check_in = date.replace(hour=np.random.randint(6, 10), minute=np.random.randint(0, 59))
            check_out = check_in + timedelta(hours=np.random.randint(7, 10))
            
            # Calculate total working hours
            total_hours = (check_out - check_in).total_seconds() / 3600
            
            # Categorize attendance
            if total_hours >= 8:
                attendance_status = "Regular"
            elif 6 <= total_hours < 8:
                attendance_status = "Partial"
            else:
                attendance_status = "Irregular"
        else:
            check_in = None
            check_out = None
            total_hours = 0
            attendance_status = "Absent"
        
        attendance_data.append({
            'employee_id': emp,
            'date': date.date(),
            'is_present': is_present,
            'check_in': check_in,
            'check_out': check_out,
            'total_hours': total_hours,
            'attendance_status': attendance_status
        })

# Create DataFrame
attendance_df = pd.DataFrame(attendance_data)

# Save to CSV
attendance_df.to_csv('../sample_data/attendance_data.csv', index=False)
print(attendance_df.head())
print("\nAttendance Summary:")
print(attendance_df['attendance_status'].value_counts())