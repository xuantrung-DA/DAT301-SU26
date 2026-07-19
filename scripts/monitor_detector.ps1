param([string]$ProjectRoot,[int]$TrainingPid,[string]$RunName,[string]$ResultsCsv,[int]$TargetEpochs=100,[int]$IntervalSeconds=30)
$statusPath=Join-Path $ProjectRoot 'runs\LIVE_TRAINING_STATUS.json'; $logPath=Join-Path $ProjectRoot 'runs\LIVE_TRAINING_STATUS.log'; $csvPath=Join-Path $ProjectRoot $ResultsCsv
while(Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue){
  $rows=@(); if(Test-Path $csvPath){try{$rows=@(Import-Csv $csvPath)}catch{$rows=@()}}
  $last=if($rows.Count){$rows[-1]}else{$null}; $g=((& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null)-split ',\s*')
  $s=[ordered]@{updated_at=(Get-Date).ToString('o');running=$true;process_id=$TrainingPid;job='detector_train';run=$RunName;epochs_completed=$rows.Count;target_epochs=$TargetEpochs;progress_percent=[math]::Round(100*$rows.Count/$TargetEpochs,2);latest_metrics=$last;gpu=@{utilization_percent=[double]$g[0];memory_used_mib=[double]$g[1];memory_total_mib=[double]$g[2];temperature_c=[double]$g[3]}}
  $s|ConvertTo-Json -Depth 6|Set-Content $statusPath -Encoding utf8
  "$(Get-Date -Format o) job=detector_train run=$RunName epoch=$($rows.Count)/$TargetEpochs gpu=$($g[0])% vram=$($g[1])MiB"|Add-Content $logPath -Encoding utf8
  Start-Sleep $IntervalSeconds
}
@{updated_at=(Get-Date).ToString('o');running=$false;job='detector_train';run=$RunName;message='Process ended; inspect results.csv/results.json.'}|ConvertTo-Json|Set-Content $statusPath -Encoding utf8
