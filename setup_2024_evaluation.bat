@echo off
REM BitMar 2024 Evaluation Pipeline Integration Script (Windows)
REM Run this script to set up BitMar for evaluation with the 2024 pipeline

echo 🔧 Setting up BitMar for 2024 BabyLM evaluation pipeline...

REM Check if evaluation-pipeline-2024 exists
if not exist "..\evaluation-pipeline-2024" (
    echo ❌ Error: evaluation-pipeline-2024 directory not found at ..\evaluation-pipeline-2024
    echo Please make sure the 2024 evaluation pipeline is cloned and available.
    exit /b 1
)

REM Copy BitMar DevBench integration file
echo 📁 Copying BitMar DevBench integration...
copy devbench_bitmar.py ..\evaluation-pipeline-2024\devbench\model_classes\bitmar.py
echo ✅ BitMar DevBench model class installed

REM Create a BitMar-specific evaluation script
echo 📝 Creating BitMar evaluation script...
(
echo @echo off
echo.
echo if "%%1"=="" ^(
echo     echo Usage: eval_bitmar.bat ^<path_to_bitmar_model^>
echo     echo Example: eval_bitmar.bat ..\BitMar\final_model_2024
echo     exit /b 1
echo ^)
echo.
echo set MODEL_PATH=%%1
echo echo 🚀 Evaluating BitMar model on 2024 BabyLM multimodal tasks...
echo echo Model path: %%MODEL_PATH%%
echo.
echo REM Change to evaluation pipeline directory
echo cd ..\evaluation-pipeline-2024
echo.
echo echo 📊 Running Winoground and VQA evaluation...
echo call eval_multimodal.sh %%MODEL_PATH%%
echo.
echo echo 🖼️ Running DevBench evaluation...
echo call eval_devbench.sh %%MODEL_PATH%% bitmar
echo.
echo echo ✅ BitMar evaluation complete!
echo echo Results are saved in the evaluation-pipeline-2024/results/ directory
) > eval_bitmar.bat

echo ✅ BitMar evaluation script created: eval_bitmar.bat

echo.
echo 🎉 BitMar integration with 2024 evaluation pipeline complete!
echo.
echo 📋 Usage Instructions:
echo 1. Train your BitMar model:
echo    python train_bitmar.py --config configs/bitmar_10epoch_memory_optimized.yaml --optimizer adamw --epochs 10 --wandb_project "bitmar-10epoch-memory-safe"
echo.
echo 2. Evaluate on 2024 pipeline (multimodal tasks):
echo    eval_bitmar.bat .\final_model_2024
echo.
echo 3. Or run individual evaluations:
echo    cd ..\evaluation-pipeline-2024
echo    eval_multimodal.sh ..\BitMar\final_model_2024
echo    eval_devbench.sh ..\BitMar\final_model_2024 bitmar
echo.
echo 📁 Model locations after training:
echo    - final_model\        : For 2025 pipeline (text-only)
echo    - final_model_2024\   : For 2024 pipeline (multimodal)
echo.
echo 🔍 Both models are identical, just placed for convenience with respective pipelines

pause
