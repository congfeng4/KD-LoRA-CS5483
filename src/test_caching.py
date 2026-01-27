import sys
sys.path.insert(0, '.')
from utils import get_tokenized_dataset, get_raw_dataset, get_tokenizer

def test_caching():
    task = 'wnli'
    dataset_path = './dataset'
    from_disk = False  # Load from Hugging Face hub (cached)
    teacher_model = './models/bert-base-uncased'
    student_model = './models/distilbert-base-uncased'
    
    print('=== First call (should miss) ===')
    tokenized1 = get_tokenized_dataset(task, teacher_model, with_indices=False, dataset_path=dataset_path, from_disk=from_disk)
    print('Tokenized shape:', tokenized1['train'].num_rows if 'train' in tokenized1 else 'N/A')
    
    print('\n=== Second call same key (should hit) ===')
    tokenized2 = get_tokenized_dataset(task, teacher_model, with_indices=False, dataset_path=dataset_path, from_disk=from_disk)
    
    print('\n=== Different tokenizer (should miss) ===')
    tokenized3 = get_tokenized_dataset(task, student_model, with_indices=True, dataset_path=dataset_path, from_disk=from_disk)
    
    print('\n=== Same tokenizer but with_indices=False (should hit) ===')
    tokenized4 = get_tokenized_dataset(task, teacher_model, with_indices=False, dataset_path=dataset_path, from_disk=from_disk)
    
    print('\n=== Raw dataset cache test ===')
    raw1 = get_raw_dataset(dataset_path, task, from_disk)
    raw2 = get_raw_dataset(dataset_path, task, from_disk)
    
    print('\n=== Tokenizer cache test ===')
    tok1 = get_tokenizer(teacher_model)
    tok2 = get_tokenizer(teacher_model)
    
    print('\nTest completed.')

if __name__ == '__main__':
    test_caching()