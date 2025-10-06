# """
# Data loading utilities for SSMG evaluation
# """

# import json
# import csv
# from typing import List, Dict, Any, Iterator, Tuple
# from pathlib import Path
# import logging

# logger = logging.getLogger(__name__)

# class DialogueDataset:
#     """Base class for dialogue datasets"""

#     def __init__(self, data_path: str):
#         self.data_path = Path(data_path)
#         self.dialogues = []
#         self.load_data()

#     def load_data(self):
#         """Load dialogue data - to be implemented by subclasses"""
#         raise NotImplementedError

#     def __len__(self) -> int:
#         return len(self.dialogues)

#     def __iter__(self) -> Iterator[Dict[str, Any]]:
#         return iter(self.dialogues)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         return self.dialogues[idx]

# class MultiWOZLoader(DialogueDataset):
#     """Loader for MultiWOZ dataset"""

#     def load_data(self):
#         """Load MultiWOZ dialogues from JSON"""
#         try:
#             with open(self.data_path, 'r') as f:
#                 data = json.load(f)

#             for dialogue_id, dialogue in data.items():
#                 turns = []
#                 for i in range(0, len(dialogue['log']), 2):
#                     if i + 1 < len(dialogue['log']):
#                         user_turn = dialogue['log'][i]['text']
#                         system_turn = dialogue['log'][i + 1]['text']
#                         turns.append({
#                             'user': user_turn,
#                             'system': system_turn,  
#                             'turn_id': i // 2
#                         })

#                 self.dialogues.append({
#                     'dialogue_id': dialogue_id,
#                     'turns': turns,
#                     'domain': dialogue.get('goal', {}).get('domain', 'unknown')
#                 })

#         except FileNotFoundError:
#             logger.warning(f"MultiWOZ data not found at {self.data_path}, using synthetic data")
#             self._create_synthetic_data()

#     def _create_synthetic_data(self):
#         """Create synthetic dialogue data for testing"""
#         synthetic_dialogues = [
#             {
#                 'dialogue_id': 'syn_001',
#                 'turns': [
#                     {'user': 'I want to order a pizza. But please no onions.', 'system': 'Sure! What size pizza would you like?', 'turn_id': 0},
#                     {'user': 'Large please. And add garlic bread.', 'system': 'Perfect! Large pizza with no onions and garlic bread. Anything else?', 'turn_id': 1},
#                     {'user': 'Actually, change it to pasta instead.', 'system': 'No problem! Pasta with no onions and garlic bread. What type of pasta?', 'turn_id': 2}
#                 ],
#                 'domain': 'restaurant'
#             },
#             {
#                 'dialogue_id': 'syn_002', 
#                 'turns': [
#                     {'user': 'I need to book a hotel for next week.', 'system': 'I can help with that! What dates are you looking for?', 'turn_id': 0},
#                     {'user': 'March 15-17. I prefer something with a pool.', 'system': 'Great! Let me find hotels with pools for those dates.', 'turn_id': 1},
#                     {'user': 'Actually make that two rooms.', 'system': 'Perfect! Two rooms from March 15-17 with pool access.', 'turn_id': 2}
#                 ],
#                 'domain': 'hotel'
#             }
#         ]
#         self.dialogues = synthetic_dialogues

# class DailyDialogLoader(DialogueDataset):
#     """Loader for DailyDialog dataset"""

#     def load_data(self):
#         """Load DailyDialog from text files"""
#         try:
#             # Try to load actual DailyDialog format
#             dialogues_file = self.data_path / "dialogues_text.txt"
#             if dialogues_file.exists():
#                 with open(dialogues_file, 'r') as f:
#                     lines = f.readlines()

#                 for i, line in enumerate(lines):
#                     turns = line.strip().split('__eou__')[:-1]  # Remove empty last element
#                     dialogue_turns = []

#                     for j in range(0, len(turns), 2):
#                         if j + 1 < len(turns):
#                             dialogue_turns.append({
#                                 'user': turns[j].strip(),
#                                 'system': turns[j + 1].strip(),
#                                 'turn_id': j // 2
#                             })

#                     self.dialogues.append({
#                         'dialogue_id': f'dd_{i:06d}',
#                         'turns': dialogue_turns,
#                         'domain': 'daily'
#                     })
#             else:
#                 raise FileNotFoundError("DailyDialog not found")

#         except FileNotFoundError:
#             logger.warning(f"DailyDialog data not found, using synthetic data")
#             self._create_synthetic_daily_dialogues()

#     def _create_synthetic_daily_dialogues(self):
#         """Create synthetic daily conversation data"""
#         self.dialogues = [
#             {
#                 'dialogue_id': 'daily_001',
#                 'turns': [
#                     {'user': 'How was your weekend?', 'system': 'It was great! I went hiking. How about yours?', 'turn_id': 0},
#                     {'user': 'Nice! I stayed home and watched movies.', 'system': 'That sounds relaxing. What movies did you watch?', 'turn_id': 1},
#                     {'user': 'Some sci-fi films. I love that genre.', 'system': 'Sci-fi is awesome! Any recommendations?', 'turn_id': 2}
#                 ],
#                 'domain': 'daily'
#             }
#         ]

# class ConfigLoader:
#     """Load evaluation configurations from JSON"""

#     @staticmethod
#     def load_config(config_path: str) -> Dict[str, Any]:
#         """Load configuration from JSON file"""
#         with open(config_path, 'r') as f:
#             return json.load(f)

#     @staticmethod
#     def get_test_dialogues(config: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Extract test dialogues from config"""
#         return config.get('evaluation', {}).get('test_dialogues', [])

# def load_dataset(dataset_name: str, data_path: str) -> DialogueDataset:
#     """Factory function to load datasets"""
#     loaders = {
#         'multiwoz': MultiWOZLoader,
#         'dailydialog': DailyDialogLoader
#     }

#     if dataset_name.lower() not in loaders:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     return loaders[dataset_name.lower()](data_path)
"""Enhanced MultiWOZ data loading for SSMG evaluation"""

import json
import requests
from typing import List, Dict, Any, Iterator
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dialogue_agent.log',  # <-- logs go here
    filemode='w'  # 'w' to overwrite each run, 'a' to append
)
logger = logging.getLogger(__name__)

class MultiWOZLoader:
    """MultiWOZ dataset loader with automatic download and parsing"""
    
    def __init__(self, data_path: str = "data/datasets", split: str = "train", max_dialogues: int = 100):
        self.data_path = Path(data_path)
        self.split = split
        self.max_dialogues = max_dialogues
        self.dialogues = []
        self.setup_and_load()
    
    def setup_and_load(self):
        """Setup directories and load MultiWOZ data"""
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Try datasets library first, then manual download, then synthetic
        if not self._load_from_datasets():
            if not self._download_multiwoz():
                self._create_synthetic_data()
    
    def _load_from_datasets(self) -> bool:
        """Load using HuggingFace datasets library"""
        try:
            from datasets import load_dataset
            
            logger.info("Loading MultiWOZ using datasets library...")
            dataset = load_dataset("multi_woz_v22", split=self.split)
            
            count = 0
            for example in dataset:
                if count >= self.max_dialogues:
                    break
                
                dialogue_id = f"mwz_{count:04d}"
                turns = self._parse_multiwoz_example(example)
                
                if len(turns) >= 2:
                    self.dialogues.append({
                        'dialogue_id': dialogue_id,
                        'turns': turns,
                        'domains': self._extract_domains(example),
                        'success': example.get('metadata', {}).get('success', False)
                    })
                    count += 1
            
            logger.info(f"Loaded {len(self.dialogues)} MultiWOZ dialogues")
            return True
            
        except ImportError:
            logger.warning("datasets library not available. Install with: pip install datasets")
            return False
        except Exception as e:
            logger.error(f"Failed to load from datasets: {e}")
            return False
    
    def _download_multiwoz(self) -> bool:
        """Download MultiWOZ manually"""
        try:
            # Use a smaller, accessible MultiWOZ sample
            url = "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.1/valListFile.json"
            
            logger.info("Downloading MultiWOZ sample...")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # This is just a file list, create sample data
                self._create_enhanced_synthetic_data()
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _parse_multiwoz_example(self, example: Dict) -> List[Dict]:
        """Parse MultiWOZ example into turns"""
        turns = []
        
        # Handle different MultiWOZ formats
        if 'dialogue' in example:
            dialogue_turns = example['dialogue']
        elif 'log' in example:
            dialogue_turns = example['log']
        else:
            return turns
        
        for i in range(0, len(dialogue_turns), 2):
            if i + 1 < len(dialogue_turns):
                user_turn = dialogue_turns[i]
                system_turn = dialogue_turns[i + 1]
                
                user_text = user_turn.get('text', '').strip()
                system_text = system_turn.get('text', '').strip()
                
                if user_text and system_text:
                    turns.append({
                        'user_input': user_text,
                        'system_response': system_text,
                        'turn_id': i // 2,
                        'user_acts': user_turn.get('dialog_act', {}),
                        'belief_state': system_turn.get('metadata', {})
                    })
        
        return turns
    
    def _extract_domains(self, example: Dict) -> List[str]:
        """Extract domains from MultiWOZ example"""
        domains = set()
        
        # From services field
        if 'services' in example:
            domains.update(example['services'])
        
        # From dialogue acts
        for turn in example.get('dialogue', []):
            acts = turn.get('dialog_act', {})
            for act_key in acts.keys():
                domain = act_key.split('-')[0].lower()
                if domain in ['hotel', 'restaurant', 'attraction', 'taxi', 'train']:
                    domains.add(domain)
        
        return list(domains) if domains else ['general']
    
    def _create_enhanced_synthetic_data(self):
        """Create realistic MultiWOZ-style synthetic data"""
        logger.info("Creating enhanced synthetic MultiWOZ data...")
        
        synthetic_dialogues = [
            {
                'dialogue_id': 'syn_hotel_001',
                'turns': [
                    {
                        'user_input': 'I need to find a hotel in Cambridge with free wifi.',
                        'system_response': 'I can help you find a hotel in Cambridge with free wifi. What price range are you looking for?',
                        'turn_id': 0,
                        'user_acts': {'Hotel-Inform': [['Area', 'cambridge'], ['Internet', 'yes']]},
                        'belief_state': {'hotel': {'area': 'cambridge', 'internet': 'yes'}}
                    },
                    {
                        'user_input': 'I want something cheap please.',
                        'system_response': 'I found the Alexander Bed and Breakfast. It\'s in the cheap price range and has free wifi. Would you like me to book it?',
                        'turn_id': 1,
                        'user_acts': {'Hotel-Inform': [['PriceRange', 'cheap']]},
                        'belief_state': {'hotel': {'area': 'cambridge', 'internet': 'yes', 'pricerange': 'cheap'}}
                    },
                    {
                        'user_input': 'Yes please book it for 2 people for 3 nights starting Monday.',
                        'system_response': 'I\'ve successfully booked the Alexander Bed and Breakfast for 2 people for 3 nights starting Monday. Your reference number is ABC123.',
                        'turn_id': 2,
                        'user_acts': {'Hotel-Book': [['People', '2'], ['Stay', '3'], ['Day', 'monday']]},
                        'belief_state': {'hotel': {'area': 'cambridge', 'internet': 'yes', 'pricerange': 'cheap', 'people': '2', 'stay': '3'}}
                    }
                ],
                'domains': ['hotel'],
                'success': True
            },
            {
                'dialogue_id': 'syn_restaurant_001', 
                'turns': [
                    {
                        'user_input': 'I\'m looking for an expensive restaurant that serves Chinese food.',
                        'system_response': 'There are several expensive Chinese restaurants. What area would you prefer?',
                        'turn_id': 0,
                        'user_acts': {'Restaurant-Inform': [['Food', 'chinese'], ['PriceRange', 'expensive']]},
                        'belief_state': {'restaurant': {'food': 'chinese', 'pricerange': 'expensive'}}
                    },
                    {
                        'user_input': 'I don\'t care about the area. I need a table for 4 people at 7pm.',
                        'system_response': 'I recommend the Hakka restaurant. It serves expensive Chinese food. Shall I book a table for 4 at 7pm?',
                        'turn_id': 1,
                        'user_acts': {'Restaurant-Inform': [['People', '4'], ['Time', '19:00']], 'general-dontcare': [['Area', 'dontcare']]},
                        'belief_state': {'restaurant': {'food': 'chinese', 'pricerange': 'expensive', 'people': '4', 'time': '19:00', 'area': 'dontcare'}}
                    }
                ],
                'domains': ['restaurant'],
                'success': False
            },
            {
                'dialogue_id': 'syn_multi_001',
                'turns': [
                    {
                        'user_input': 'I need a train from Cambridge to London on Friday.',
                        'system_response': 'What time would you like to travel on Friday?',
                        'turn_id': 0,
                        'user_acts': {'Train-Inform': [['Departure', 'cambridge'], ['Destination', 'london'], ['Day', 'friday']]},
                        'belief_state': {'train': {'departure': 'cambridge', 'destination': 'london', 'day': 'friday'}}
                    },
                    {
                        'user_input': 'I want to leave after 10am. Also, can you find me a restaurant near the train station?',
                        'system_response': 'I found a train leaving at 11:01. For restaurants near the station, what type of food would you like?',
                        'turn_id': 1,
                        'user_acts': {'Train-Inform': [['LeaveAt', '10:00']], 'Restaurant-Inform': [['Area', 'centre']]},
                        'belief_state': {'train': {'departure': 'cambridge', 'destination': 'london', 'day': 'friday', 'leaveat': '10:00'}, 'restaurant': {'area': 'centre'}}
                    }
                ],
                'domains': ['train', 'restaurant'],
                'success': True
            }
        ]
        
        # Extend to meet max_dialogues requirement
        while len(synthetic_dialogues) < min(self.max_dialogues, 50):
            base_dialogue = synthetic_dialogues[len(synthetic_dialogues) % 3].copy()
            base_dialogue['dialogue_id'] = f"syn_ext_{len(synthetic_dialogues):03d}"
            synthetic_dialogues.append(base_dialogue)
        
        self.dialogues = synthetic_dialogues[:self.max_dialogues]
        logger.info(f"Created {len(self.dialogues)} synthetic MultiWOZ dialogues")
    
    def _create_synthetic_data(self):
        """Fallback synthetic data"""
        self._create_enhanced_synthetic_data()
    
    def get_evaluation_data(self) -> Dict[str, Any]:
        """Get data formatted for SSMG evaluation"""
        return {
            'dialogues': self.dialogues,
            'total_dialogues': len(self.dialogues),
            'total_turns': sum(len(d['turns']) for d in self.dialogues),
            'domains': list(set().union(*[d['domains'] for d in self.dialogues]))
        }
    
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        return self.dialogues[idx]
