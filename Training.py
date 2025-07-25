import os
import TrainingConfig
import bofh_trainer

TRAINING_CONFIG = TrainingConfig.TrainerConfig()

def main():
    # setup checkpointing
    os.makedirs(TRAINING_CONFIG.checkpoint_dir, exist_ok=True)
    
    try: 
        trainer = bofh_trainer.BOFHTrainer(TRAINING_CONFIG)    
        trainer.setup()
        trainer.train()
    except Exception as e:
        print(f"TRAINING FAILURE : {e}")
        
if __name__ == "__main__":
    main()
    