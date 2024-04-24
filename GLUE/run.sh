python cli.py --relation_id=CoLA --num_class=2 --batch_size=32 --lr=5e-5 --a_lr=5e-5 --p_lr=4e-3 --cuda_device=cuda:0 --gan_epochs=2 --a_epochs=30 --p_epochs=30
python cli.py --relation_id=RTE --num_class=2 --batch_size=32 --lr=5e-5 --a_lr=5e-5 --p_lr=4e-3 --cuda_device=cuda:0 --gan_epochs=2 --a_epochs=30 --p_epochs=30
python cli.py --relation_id=STSB --num_class=1 --batch_size=32 --lr=5e-5 --a_lr=5e-5 --p_lr=4e-3 --cuda_device=cuda:0 --gan_epochs=2 --a_epochs=30 --p_epochs=30
