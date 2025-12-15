@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Generate per-p training datasets data\train_p{p}.txt using the same sampler settings as pipeline.cmd
REM Usage: prepare_train_data.cmd [POVM] [L] [FULL_N]
REM Defaults mirror pipeline.cmd
set "POVM=Tetra"
set "L=2"
set "FULL_N=50"
set "PY=python"
set "TENSOR=D:\CSE550_final_project_quantum_state_learning\data\training_data_1D_TFIM_model\Matrix_product_state\tensor.txt"
set "NSAMPLES=60000"
REM Space-separated list of noise strengths to cover
set "P_LIST=0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

if not "%~1"=="" set "POVM=%~1"
if not "%~2"=="" set "L=%~2"
if not "%~3"=="" set "FULL_N=%~3"
if not "%~4"=="" set "P_LIST=%~4"

if not exist "D:\CSE550_final_project_quantum_state_learning\data" mkdir "D:\CSE550_final_project_quantum_state_learning\data"

for %%P in (%P_LIST%) do (
  set "CUR_P=%%P"
  echo ================================================
  echo Generating train_p!CUR_P!.txt ^(POVM=%POVM%, L=%L%, FULL_N=%FULL_N%^)
  echo ================================================

  if exist "D:\CSE550_final_project_quantum_state_learning\train.txt" del /f /q "D:\CSE550_final_project_quantum_state_learning\train.txt"
  if exist "D:\CSE550_final_project_quantum_state_learning\data.txt" del /f /q "D:\CSE550_final_project_quantum_state_learning\data.txt"
  if exist "D:\CSE550_final_project_quantum_state_learning\data\train.txt" del /f /q "D:\CSE550_final_project_quantum_state_learning\data\train.txt"

  call "%PY%" "D:\CSE550_final_project_quantum_state_learning\MPS_POVM_sampler\noisygeneration.py" %POVM% %FULL_N% "%TENSOR%" !CUR_P! %NSAMPLES% %L%
  if errorlevel 1 (
    echo Sampler failed for p=!CUR_P! with errorlevel !errorlevel!
    exit /b !errorlevel!
  )

  if not exist "D:\CSE550_final_project_quantum_state_learning\train.txt" (
    echo Expected train.txt not found after sampling for p=!CUR_P!
    exit /b 1
  )

  move /Y "D:\CSE550_final_project_quantum_state_learning\train.txt" "D:\CSE550_final_project_quantum_state_learning\data\train.txt" >nul
  copy /Y "D:\CSE550_final_project_quantum_state_learning\data\train.txt" "D:\CSE550_final_project_quantum_state_learning\data\train_p!CUR_P!.txt" >nul
  echo Saved data\train_p!CUR_P!.txt
)

echo === All datasets generated successfully ===
exit /b 0
