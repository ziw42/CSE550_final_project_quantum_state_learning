@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Sampling -> RNN training -> Bell evaluation pipeline
REM Usage: pipeline_rnn.cmd [POVM] [p] [QUBITS] [FULL_N] [LATENT] [HIDDEN]
REM Defaults:
set "POVM=Tetra"
set "P=0.1"
set "PY=python"
set "FULL_N=50"
set "TENSOR=D:\CSE550_final_project_quantum_state_learning\data\training_data_1D_TFIM_model\Matrix_product_state\tensor.txt"
set "NSAMPLES=60000"
set "L=2"
set "LATENT=100"
set "HIDDEN=100"
set "LR=0.001"
set "RNN_TRAIN_SAMPLES=0"
set "RNN_EVAL_SAMPLES=60000"
set "EPOCHS=60"

if not "%~1"=="" set "POVM=%~1"
if not "%~2"=="" set "P=%~2"
if not "%~3"=="" set "L=%~3"
if not "%~4"=="" set "FULL_N=%~4"
if not "%~5"=="" set "LATENT=%~5"
if not "%~6"=="" set "HIDDEN=%~6"

echo === RNN Pipeline Configuration ===
echo POVM=%POVM%
echo p=%P%
echo FULL_N=%FULL_N% L=%L% K=?, LATENT=%LATENT% HIDDEN=%HIDDEN%
echo ===================================

REM Step 1: clean old train/data files
if exist "D:\CSE550_final_project_quantum_state_learning\train.txt" del /f /q "D:\CSE550_final_project_quantum_state_learning\train.txt"
if exist "D:\CSE550_final_project_quantum_state_learning\data.txt" del /f /q "D:\CSE550_final_project_quantum_state_learning\data.txt"
if not exist "D:\CSE550_final_project_quantum_state_learning\data" mkdir "D:\CSE550_final_project_quantum_state_learning\data"

REM Step 2: sampling
set "STEP_LABEL=[1/3]"
echo %STEP_LABEL% Sampling POVM measurements ...
call "%PY%" "D:\CSE550_final_project_quantum_state_learning\MPS_POVM_sampler\noisygeneration.py" %POVM% %FULL_N% "%TENSOR%" %P% %NSAMPLES% %L%
if errorlevel 1 goto :err
echo POVM=%POVM%
echo p=%P%
echo FULL_N=%FULL_N% L=%L% K=%K% LATENT=%LATENT% HIDDEN=%HIDDEN%

set "TRAIN_FILE=D:\CSE550_final_project_quantum_state_learning\data\train_p%P%.txt"
if exist "%TRAIN_FILE%" del /f /q "%TRAIN_FILE%"
if exist "D:\CSE550_final_project_quantum_state_learning\train.txt" move /Y "D:\CSE550_final_project_quantum_state_learning\train.txt" "%TRAIN_FILE%" >nul

REM Determine K from POVM
set "K=4"
if /I "%POVM%"=="Trine" set "K=3"
if /I "%POVM%"=="Pauli" set "K=6"
if /I "%POVM%"=="4Pauli" set "K=4"

echo [2/3] Training RNN ...
call "%PY%" "D:\CSE550_final_project_quantum_state_learning\models\RNN\train_rnn.py" --data "%TRAIN_FILE%" --L %L% --K %K% --latent %LATENT% --hidden %HIDDEN% --samples %RNN_TRAIN_SAMPLES% --p %P% --lr %LR% --epochs %EPOCHS%
if errorlevel 1 goto :err

set "RNN_DIR=D:\CSE550_final_project_quantum_state_learning\RNN_parameters\L%L%_K%K%_latent%LATENT%_hid%HIDDEN%"
set "RNN_CKPT="
if exist "%RNN_DIR%" (
  for /f "delims=" %%f in ('dir /b /o-d "%RNN_DIR%\model-*.index"') do (
    if not defined RNN_CKPT (

set "RNN_SAMPLE_OUT=D:\CSE550_final_project_quantum_state_learning\samples\rnn_samples_p%P%.txt"

echo [3/3] Evaluating RNN vs Bell distribution ...
call "%PY%" "D:\CSE550_final_project_quantum_state_learning\evaluation\score_rnn.py" --checkpoint "%RNN_CKPT%" --data "%TRAIN_FILE%" --tensor "%TENSOR%" --povm %POVM% --p %P% --full-n %FULL_N% --K %K% --L %L% --latent %LATENT% --hidden %HIDDEN% --num-samples %RNN_EVAL_SAMPLES% --samples-out "%RNN_SAMPLE_OUT%"
if errorlevel 1 goto :err

go to :post_eval

:no_ckpt
echo WARNING: No RNN checkpoints found in %RNN_DIR%.

:post_eval
if errorlevel 1 goto :err

echo === RNN Pipeline completed successfully ===
goto :eof

:err
echo === pipeline_rnn failed with errorlevel %errorlevel% ===
exit /b %errorlevel%
*** End Patch