import sys
sys.path.insert(0, '.')
from addict import Addict

def test_loop_caching():
    """Simulate hyperparameter loops to verify caching."""
    base_args = {
        'teacher_model_name': './models/bert-base-uncased',
        'student_model_name': './models/distilbert-base-uncased',
        'dataset_path': './dataset',
        'dir_name': './results',
        'teacher_learning_rate': 5e-5,
        'student_learning_rate': 5e-4,
        'num_train_epochs': 1,
        'rank': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'peft': 'lora',
        'task': 'wnli',
        'train_batch_size': 32,
        'eval_batch_size': 32,
        'weight_decay': 0.01,
        'from_disk': 0,  # load from hub cache
        'seed': 42,
    }
    
    # Simulate two iterations with same task, same tokenizers
    print('=== Iteration 1 ===')
    from BERT_Distill_LoRA import BertDistillPipeline
    pipe1 = BertDistillPipeline(**base_args)
    teacher_dataset = pipe1.load_dataset()
    tokenized_teacher1 = pipe1.tokenize_teacher_dataset(teacher_dataset)
    tokenized_student1 = pipe1.tokenize_student_dataset(teacher_dataset)
    print(f'Teacher tokenized: {tokenized_teacher1}')
    print(f'Student tokenized: {tokenized_student1}')
    
    print('\n=== Iteration 2 (same hyperparameters) ===')
    pipe2 = BertDistillPipeline(**base_args)
    teacher_dataset2 = pipe2.load_dataset()
    tokenized_teacher2 = pipe2.tokenize_teacher_dataset(teacher_dataset2)
    tokenized_student2 = pipe2.tokenize_student_dataset(teacher_dataset2)
    
    # Verify that the same dataset objects are returned (cached)
    assert tokenized_teacher1 is tokenized_teacher2, "Teacher tokenized dataset not cached"
    assert tokenized_student1 is tokenized_student2, "Student tokenized dataset not cached"
    print('Caching verified: same objects returned')
    
    # Simulate different task (should miss)
    print('\n=== Iteration 3 (different task) ===')
    args3 = base_args.copy()
    args3['task'] = 'sst2'
    pipe3 = BertDistillPipeline(**args3)
    teacher_dataset3 = pipe3.load_dataset()
    tokenized_teacher3 = pipe3.tokenize_teacher_dataset(teacher_dataset3)
    # Should be different object
    assert tokenized_teacher1 is not tokenized_teacher3, "Different task should not cache"
    print('Different task correctly missed cache')
    
    print('\nAll caching tests passed.')

if __name__ == '__main__':
    test_loop_caching()