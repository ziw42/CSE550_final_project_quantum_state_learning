@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Sampling -> evaluation pipeline focused on disambiguating data files
REM Edit parameters below for different settings
set "POVM=Tetra"
set "P=0.9"
set "PY=python"
set "FULL_N=50"
set "TENSOR=D:\CSE550_final_project_quantum_state_learning\data\training_data_1D_TFIM_model\Matrix_product_state\tensor.txt"
set "NSAMPLES=60000"
set "L=2"

if not "%~1"=="" set "POVM=%~1"
if not "%~2"=="" set "P=%~2"
if not "%~3"=="" set "L=%~3"
if not "%~4"=="" set "FULL_N=%~4"

echo === Configuration ===
echo POVM=%POVM%
echo p=%P%
echo FULL_N=%FULL_N% TENSOR=%TENSOR% NSAMPLES=%NSAMPLES% L=%L%
echo =====================

REM Step 1: delete stale root-level train.txt and data.txt
if exist "D:\CSE550_final_project_quantum_state_learning\train.txt" del /f /q "D:\CSE550_final_project_quantum_state_learning\train.txt"
if exist "D:\CSE550_final_project_quantum_state_learning\data.txt" del /f /q "D:\CSE550_final_project_quantum_state_learning\data.txt"

REM Ensure data directory exists
if not exist "D:\CSE550_final_project_quantum_state_learning\data" mkdir "D:\CSE550_final_project_quantum_state_learning\data"

echo [1/3] Sampling POVM measurements to train.txt
call "%PY%" "D:\CSE550_final_project_quantum_state_learning\MPS_POVM_sampler\noisygeneration.py" %POVM% %FULL_N% "%TENSOR%" %P% %NSAMPLES% %L%
if errorlevel 1 goto :err

REM Step 3: replace data\train.txt with freshly generated root-level train.txt
if exist "D:\CSE550_final_project_quantum_state_learning\data\train.txt" del /f /q "D:\CSE550_final_project_quantum_state_learning\data\train.txt"
if exist "D:\CSE550_final_project_quantum_state_learning\train.txt" move /Y "D:\CSE550_final_project_quantum_state_learning\train.txt" "D:\CSE550_final_project_quantum_state_learning\data\train.txt" >nul

REM Infer K from POVM for evaluation block checks and per-site scaling
set "K=4"
if /I "%POVM%"=="Trine"  set "K=3"
if /I "%POVM%"=="Pauli"  set "K=6"
if /I "%POVM%"=="4Pauli" set "K=4"

echo [2/3] Training RBM to match current data shape 
call "%PY%" "D:\CSE550_final_project_quantum_state_learning\models\RBM\train_rbm.py" --p %P% --L %L% --num_state_vis %K% --cd
if errorlevel 1 goto :err

echo [3/3] Evaluating RBM against Bell distribution
REM Pick latest in RBM_parameters_p{p}_L{L}_K{K}
set "PARAM_DIR=D:\CSE550_final_project_quantum_state_learning\RBM_parameters_p%P%_L%L%_K%K%"
set "PARAMS="

if exist "%PARAM_DIR%" (
  REM Take most recent .npz by modified time (first in /o-d)
  for /f "delims=" %%f in ('dir /b /o-d "%PARAM_DIR%\*.npz"') do (
    if not defined PARAMS (
      set "PARAMS=%PARAM_DIR%\%%f"
    )
  )
)

if "%PARAMS%"=="" goto :no_params

echo Using parameters: %PARAMS%
call "%PY%" "D:\CSE550_final_project_quantum_state_learning\evaluation\score_rbm.py" --params "%PARAMS%" --data "D:\CSE550_final_project_quantum_state_learning\data\train.txt" --K %K% --tensor "%TENSOR%" --povm %POVM% --full-n %FULL_N% --p %P% --L %L%
if errorlevel 1 goto :err

set "P_TAG=%P%"
if exist "D:\CSE550_final_project_quantum_state_learning\data\train.txt" (
  echo Copying data\train.txt to data\train_p%P_TAG%.txt for plotting consistency...
  copy /Y "D:\CSE550_final_project_quantum_state_learning\data\train.txt" "D:\CSE550_final_project_quantum_state_learning\data\train_p%P_TAG%.txt" >nul
)
goto :eof

:err
echo === Pipeline failed with errorlevel %errorlevel% ===
exit /b %errorlevel%
