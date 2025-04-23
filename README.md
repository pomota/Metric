1. Clone repository  

2. Add Files  
- result/input/input.csv
- result/ground_truth: For score calculation (Template과 column 동일하게 유지!!)  
- .env: Containing API_KEY for LLM
  
input.csv Columns: id,report  
  
Output file default path: "/mnt/nas125/forGPU2/jyseo/1.Report_Metric/final/metric/result/output"  
  
3. Run "python -m metric.py"  