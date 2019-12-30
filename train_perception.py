from scripts.perception.perception_trainer import PerceptionTrainer

data_path = "/home/feiyang/Downloads/data/town1"
trainer = PerceptionTrainer(data_path=data_path, split='training', epoch=100)
model = trainer.fit()
trainer.save_model("./models/perception/")