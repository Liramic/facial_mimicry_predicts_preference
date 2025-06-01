from Analysis.run_analysis import run_analysis
from Analysis.convert_excel_to_r import export_reading_to_r, export_listening_to_r
from General.init import *

if __name__ == '__main__':
   rms = 50
   ds = 5
   window = 3000
   max_lag = 3000
   action = "listening" # or "reading"
   
   csv_out_path = fr"{action}_rms_{rms}_ds_{ds}_window_{window}_max_lag_{max_lag}_tInc5.csv"
   df = run_analysis(rms, ds, window, max_lag, csv_out_path=None, action=action)
   df.to_csv(csv_out_path, index=False)

   if action == "reading":
       export_reading_to_r(csv_out_path)
   elif action == "listening":
      export_listening_to_r(csv_out_path)
