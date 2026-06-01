# README

# The implementation of "EntroRec: Mitigating the Matthew Effect via  Entropy Enhancement in Generative Recommendation" .

## File Description
- `sft.py`: the SFT code
- `EntroRec.py`: the EntroRec code
- `EntroRec_trainer.py`: the GRPO trainer tailored for Generative Recommendation

The training instructions can be seen in `EntroRec.sh` and `train.sh`, while the evaluation instructions are in `evaluate.sh`.

## Quickstart
- Create a virtual Python environment.
`bash`
conda create -n EntroRec
- Install required packages.
`bash`
pip install -r requirements.txt
- Execute the EntroRec the training bash.
bash EntroRec.sh
- Run the evaluation bash.
bash evaluation.sh	
