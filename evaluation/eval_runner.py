"""
Main evaluation runner for SSMG experiments
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import sys
sys.path.append('src')
sys.path.append('data')
# sys.path.append(str(Path(__file__).resolve().parent.parent))

from ssmg.integration import SSMGDialogueAgent, GroqAPIInterface #type: ignore
from ssmg.summarizer import SummaryConfig #type: ignore
from evaluation.baselines import FullHistoryBaseline, SlidingWindowBaseline, RAGBaseline
from evaluation.metrics import SSMGEvaluator, EvaluationMetrics
from data.loaders import  ConfigLoader
from data.loaders import MultiWOZLoader

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='evaluation_baseline.log',  
    filemode='w'  
)
logger = logging.getLogger(__name__)




# List of all unique true intents
true_intents = [
    "inform",
    "no_offer",
    "recommend",
    "request",
    "select",
    "book",
    "no_book",
    "offer_book",
    "offer_booked",
    "bye",
    "greet",
    "request_more",
    "thank",
    "welcome"
]

# Mapping from extracted intent label to a list of possible true intents, with reasoning as comments
extracted_to_true_intent = {
    # "restaurant_reviews": ["inform"] - User is asking for reviews, so system provides information.
    "restaurant_reviews": ["inform"],

    # "nutrition_info": ["inform"] - User requests nutrition info, system provides facts.
    "nutrition_info": ["inform"],

    # "account_blocked": ["inform"] - User is told about account status, system informs.
    "account_blocked": ["inform"],

    # "oil_change_how": ["request", "inform"] - User may ask how to do oil change (request), or system may explain (inform).
    "oil_change_how": ["request", "inform"],

    # "time": ["request", "inform"] - User asks for time (request), system gives info (inform).
    "time": ["request", "inform"],

    # "weather": ["request", "inform"] - User asks about weather (request), system provides info (inform).
    "weather": ["request", "inform"],

    # "redeem_rewards": ["inform", "request"] - User may ask how to redeem (request) or system may confirm redemption (inform).
    "redeem_rewards": ["inform", "request"],

    # "interest_rate": ["request", "inform"] - User asks for rates (request), system provides info (inform).
    "interest_rate": ["request", "inform"],

    # "gas_type": ["request", "inform"] - User asks which gas to use (request), system tells (inform).
    "gas_type": ["request", "inform"],

    # "accept_reservations": ["request", "inform"] - User asks if reservations are accepted (request), system answers (inform).
    "accept_reservations": ["request", "inform"],

    # "smart_home": ["inform", "request"] - User may ask about smart home (request), or system provides info (inform).
    "smart_home": ["inform", "request"],

    # "user_name": ["request", "inform"] - User asks for their username (request), system provides (inform).
    "user_name": ["request", "inform"],

    # "report_lost_card": ["inform", "request"] - User may request to report (request), system confirms (inform).
    "report_lost_card": ["inform", "request"],

    # "repeat": ["request_more"] - User asks to repeat, system gives more info.
    "repeat": ["request_more"],

    # "whisper_mode": ["inform", "request"] - User may request whisper mode (request), system confirms (inform).
    "whisper_mode": ["inform", "request"],

    # "what_are_your_hobbies": ["request"] - User asks about system's hobbies.
    "what_are_your_hobbies": ["request"],

    # "order": ["book", "inform"] - User places an order (book), system confirms (inform).
    "order": ["book", "inform"],

    # "jump_start": ["request", "inform"] - User asks for jump start (request), system provides info (inform).
    "jump_start": ["request", "inform"],

    # "schedule_meeting": ["book", "request"] - User asks to schedule (book), or how to schedule (request).
    "schedule_meeting": ["book", "request"],

    # "meeting_schedule": ["inform", "request"] - User asks for schedule (request), system provides (inform).
    "meeting_schedule": ["inform", "request"],

    # "freeze_account": ["inform", "request"] - User requests freeze (request), system confirms (inform).
    "freeze_account": ["inform", "request"],

    # "what_song": ["request"] - User asks what song is playing.
    "what_song": ["request"],

    # "meaning_of_life": ["request"] - User asks for meaning.
    "meaning_of_life": ["request"],

    # "restaurant_reservation": ["book", "request"] - User requests reservation (book), or asks about it (request).
    "restaurant_reservation": ["book", "request"],

    # "traffic": ["request", "inform"] - User asks about traffic (request), system provides info (inform).
    "traffic": ["request", "inform"],

    # "make_call": ["request", "book"] - User asks to make a call (request), system initiates (book).
    "make_call": ["request", "book"],

    # "text": ["request", "book"] - User asks to send a text (request), system sends (book).
    "text": ["request", "book"],

    # "bill_balance": ["request", "inform"] - User asks for balance (request), system provides (inform).
    "bill_balance": ["request", "inform"],

    # "improve_credit_score": ["request", "inform"] - User asks how to improve (request), system explains (inform).
    "improve_credit_score": ["request", "inform"],

    # "change_language": ["request", "inform"] - User requests language change (request), system confirms (inform).
    "change_language": ["request", "inform"],

    # "no": ["no_offer"] - User declines an offer.
    "no": ["no_offer"],

    # "measurement_conversion": ["request"] - User asks to convert units.
    "measurement_conversion": ["request"],

    # "timer": ["request", "book"] - User asks to set timer (request), system sets (book).
    "timer": ["request", "book"],

    # "flip_coin": ["request"] - User asks to flip a coin.
    "flip_coin": ["request"],

    # "do_you_have_pets": ["request"] - User asks about system's pets.
    "do_you_have_pets": ["request"],

    # "balance": ["request", "inform"] - User asks for balance (request), system provides (inform).
    "balance": ["request", "inform"],

    # "tell_joke": ["request"] - User asks for a joke.
    "tell_joke": ["request"],

    # "last_maintenance": ["inform", "request"] - User asks when last maintenance was (request), system tells (inform).
    "last_maintenance": ["inform", "request"],

    # "exchange_rate": ["request", "inform"] - User asks for exchange rate (request), system provides (inform).
    "exchange_rate": ["request", "inform"],

    # "uber": ["book", "request"] - User requests Uber (book), or info about Uber (request).
    "uber": ["book", "request"],

    # "car_rental": ["book", "request"] - User requests rental (book), or info (request).
    "car_rental": ["book", "request"],

    # "credit_limit": ["request", "inform"] - User asks for limit (request), system provides (inform).
    "credit_limit": ["request", "inform"],

    # "oos": ["unknown"] - Out of scope.
    "oos": ["unknown"],

    # "shopping_list": ["inform", "request"] - User asks about list (request), system provides (inform).
    "shopping_list": ["inform", "request"],

    # "expiration_date": ["request", "inform"] - User asks for expiration (request), system provides (inform).
    "expiration_date": ["request", "inform"],

    # "routing": ["request", "inform"] - User asks for routing info (request), system provides (inform).
    "routing": ["request", "inform"],

    # "meal_suggestion": ["recommend", "inform"] - User asks for suggestion (recommend), system provides (inform).
    "meal_suggestion": ["recommend", "inform"],

    # "tire_change": ["request", "inform"] - User asks how to change tire (request), system explains (inform).
    "tire_change": ["request", "inform"],

    # "todo_list": ["inform", "request"] - User asks about to-do list (request), system provides (inform).
    "todo_list": ["inform", "request"],

    # "card_declined": ["inform"] - System informs user of declined card.
    "card_declined": ["inform"],

    # "rewards_balance": ["request", "inform"] - User asks for rewards (request), system provides (inform).
    "rewards_balance": ["request", "inform"],

    # "change_accent": ["request", "inform"] - User requests accent change (request), system confirms (inform).
    "change_accent": ["request", "inform"],

    # "vaccines": ["request", "inform"] - User asks about vaccines (request), system provides (inform).
    "vaccines": ["request", "inform"],

    # "reminder_update": ["request", "inform"] - User asks to update reminder (request), system confirms (inform).
    "reminder_update": ["request", "inform"],

    # "food_last": ["request", "inform"] - User asks when food was last eaten (request), system provides (inform).
    "food_last": ["request", "inform"],

    # "change_ai_name": ["request", "inform"] - User requests name change (request), system confirms (inform).
    "change_ai_name": ["request", "inform"],

    # "bill_due": ["request", "inform"] - User asks when bill is due (request), system provides (inform).
    "bill_due": ["request", "inform"],

    # "who_do_you_work_for": ["request"] - User asks about system's employer.
    "who_do_you_work_for": ["request"],

    # "share_location": ["request", "inform"] - User asks to share location (request), system confirms (inform).
    "share_location": ["request", "inform"],

    # "international_visa": ["request", "inform"] - User asks about visa (request), system provides (inform).
    "international_visa": ["request", "inform"],

    # "calendar": ["inform", "request"] - User asks about calendar (request), system provides (inform).
    "calendar": ["inform", "request"],

    # "translate": ["request"] - User asks to translate.
    "translate": ["request"],

    # "carry_on": ["request", "inform"] - User asks about carry-on (request), system provides (inform).
    "carry_on": ["request", "inform"],

    # "book_flight": ["book", "request"] - User requests to book flight (book), or info (request).
    "book_flight": ["book", "request"],

    # "insurance_change": ["request", "inform"] - User requests insurance change (request), system confirms (inform).
    "insurance_change": ["request", "inform"],

    # "todo_list_update": ["request", "inform"] - User asks to update to-do list (request), system confirms (inform).
    "todo_list_update": ["request", "inform"],

    # "timezone": ["request", "inform"] - User asks for timezone (request), system provides (inform).
    "timezone": ["request", "inform"],

    # "cancel_reservation": ["no_book"] - User cancels reservation.
    "cancel_reservation": ["no_book"],

    # "transactions": ["inform", "request"] - User asks about transactions (request), system provides (inform).
    "transactions": ["inform", "request"],

    # "credit_score": ["request", "inform"] - User asks for credit score (request), system provides (inform).
    "credit_score": ["request", "inform"],

    # "report_fraud": ["inform", "request"] - User requests to report fraud (request), system confirms (inform).
    "report_fraud": ["inform", "request"],

    # "spending_history": ["inform", "request"] - User asks about history (request), system provides (inform).
    "spending_history": ["inform", "request"],

    # "directions": ["request", "inform"] - User asks for directions (request), system provides (inform).
    "directions": ["request", "inform"],

    # "spelling": ["request"] - User asks for spelling.
    "spelling": ["request"],

    # "insurance": ["request", "inform"] - User asks about insurance (request), system provides (inform).
    "insurance": ["request", "inform"],

    # "what_is_your_name": ["greet", "request"] - User greets or asks for name.
    "what_is_your_name": ["greet", "request"],

    # "reminder": ["request", "inform"] - User asks to set reminder (request), system confirms (inform).
    "reminder": ["request", "inform"],

    # "where_are_you_from": ["request"] - User asks about system's origin.
    "where_are_you_from": ["request"],

    # "distance": ["request", "inform"] - User asks for distance (request), system provides (inform).
    "distance": ["request", "inform"],

    # "payday": ["request", "inform"] - User asks about payday (request), system provides (inform).
    "payday": ["request", "inform"],

    # "flight_status": ["request", "inform"] - User asks for flight status (request), system provides (inform).
    "flight_status": ["request", "inform"],

    # "find_phone": ["request", "inform"] - User asks to find phone (request), system confirms (inform).
    "find_phone": ["request", "inform"],

    # "greeting": ["greet"] - User greets.
    "greeting": ["greet"],

    # "alarm": ["request", "book"] - User asks to set alarm (request), system sets (book).
    "alarm": ["request", "book"],

    # "order_status": ["request", "inform"] - User asks for order status (request), system provides (inform).
    "order_status": ["request", "inform"],

    # "confirm_reservation": ["offer_booked", "inform"] - System confirms reservation (offer_booked), or informs.
    "confirm_reservation": ["offer_booked", "inform"],

    # "cook_time": ["request", "inform"] - User asks for cook time (request), system provides (inform).
    "cook_time": ["request", "inform"],

    # "damaged_card": ["inform", "request"] - User reports damaged card (request), system confirms (inform).
    "damaged_card": ["inform", "request"],

    # "reset_settings": ["request", "inform"] - User asks to reset (request), system confirms (inform).
    "reset_settings": ["request", "inform"],

    # "pin_change": ["request", "inform"] - User asks to change PIN (request), system confirms (inform).
    "pin_change": ["request", "inform"],

    # "replacement_card_duration": ["request", "inform"] - User asks how long for replacement (request), system provides (inform).
    "replacement_card_duration": ["request", "inform"],

    # "new_card": ["request", "inform"] - User asks for new card (request), system confirms (inform).
    "new_card": ["request", "inform"],

    # "roll_dice": ["request"] - User asks to roll dice.
    "roll_dice": ["request"],

    # "income": ["request", "inform"] - User asks about income (request), system provides (inform).
    "income": ["request", "inform"],

    # "taxes": ["request", "inform"] - User asks about taxes (request), system provides (inform).
    "taxes": ["request", "inform"],

    # "date": ["request", "inform"] - User asks for date (request), system provides (inform).
    "date": ["request", "inform"],

    # "who_made_you": ["request"] - User asks about system's creator.
    "who_made_you": ["request"],

    # "pto_request": ["request", "book"] - User requests PTO (request/book).
    "pto_request": ["request", "book"],

    # "tire_pressure": ["request", "inform"] - User asks about tire pressure (request), system provides (inform).
    "tire_pressure": ["request", "inform"],

    # "how_old_are_you": ["request"] - User asks about system's age.
    "how_old_are_you": ["request"],

    # "rollover_401k": ["request", "inform"] - User asks about 401k rollover (request), system provides (inform).
    "rollover_401k": ["request", "inform"],

    # "pto_request_status": ["request", "inform"] - User asks about PTO status (request), system provides (inform).
    "pto_request_status": ["request", "inform"],

    # "how_busy": ["request", "inform"] - User asks how busy system is (request), system provides (inform).
    "how_busy": ["request", "inform"],

    # "application_status": ["request", "inform"] - User asks about application (request), system provides (inform).
    "application_status": ["request", "inform"],

    # "recipe": ["request", "inform"] - User asks for recipe (request), system provides (inform).
    "recipe": ["request", "inform"],

    # "calendar_update": ["request", "inform"] - User asks to update calendar (request), system confirms (inform).
    "calendar_update": ["request", "inform"],

    # "play_music": ["request", "book"] - User asks to play music (request), system plays (book).
    "play_music": ["request", "book"],

    # "yes": ["inform"] - User affirms.
    "yes": ["inform"],

    # "direct_deposit": ["request", "inform"] - User asks about direct deposit (request), system provides (inform).
    "direct_deposit": ["request", "inform"],

    # "credit_limit_change": ["request", "inform"] - User asks to change limit (request), system confirms (inform).
    "credit_limit_change": ["request", "inform"],

    # "gas": ["request", "inform"] - User asks about gas (request), system provides (inform).
    "gas": ["request", "inform"],

    # "pay_bill": ["request", "book"] - User asks to pay bill (request), system pays (book).
    "pay_bill": ["request", "book"],

    # "ingredients_list": ["request", "inform"] - User asks for ingredients (request), system provides (inform).
    "ingredients_list": ["request", "inform"],

    # "lost_luggage": ["inform", "request"] - User reports lost luggage (request), system confirms (inform).
    "lost_luggage": ["inform", "request"],

    # "goodbye": ["bye"] - User says goodbye.
    "goodbye": ["bye"],

    # "what_can_i_ask_you": ["request"] - User asks about system capabilities.
    "what_can_i_ask_you": ["request"],

    # "book_hotel": ["book", "request"] - User requests to book hotel (book), or info (request).
    "book_hotel": ["book", "request"],

    # "are_you_a_bot": ["request"] - User asks if system is a bot.
    "are_you_a_bot": ["request"],

    # "next_song": ["request", "book"] - User asks to play next song (request), system plays (book).
    "next_song": ["request", "book"],

    # "change_speed": ["request", "inform"] - User asks to change speed (request), system confirms (inform).
    "change_speed": ["request", "inform"],

    # "plug_type": ["request", "inform"] - User asks about plug type (request), system provides (inform).
    "plug_type": ["request", "inform"],

    # "maybe": ["inform"] - User is unsure, system acknowledges.
    "maybe": ["inform"],

    # "w2": ["request", "inform"] - User asks about W2 (request), system provides (inform).
    "w2": ["request", "inform"],

    # "oil_change_when": ["request", "inform"] - User asks when oil change is due (request), system provides (inform).
    "oil_change_when": ["request", "inform"],

    # "thank_you": ["thank"] - User thanks system.
    "thank_you": ["thank"],

    # "shopping_list_update": ["request", "inform"] - User asks to update list (request), system confirms (inform).
    "shopping_list_update": ["request", "inform"],

    # "pto_balance": ["request", "inform"] - User asks for PTO balance (request), system provides (inform).
    "pto_balance": ["request", "inform"],

    # "order_checks": ["request", "book"] - User asks to order checks (request), system orders (book).
    "order_checks": ["request", "book"],

    # "travel_alert": ["request", "inform"] - User asks to set travel alert (request), system confirms (inform).
    "travel_alert": ["request", "inform"],

    # "fun_fact": ["request", "inform"] - User asks for fun fact (request), system provides (inform).
    "fun_fact": ["request", "inform"],

    # "sync_device": ["request", "inform"] - User asks to sync device (request), system confirms (inform).
    "sync_device": ["request", "inform"],

    # "schedule_maintenance": ["book", "request"] - User asks to schedule maintenance (book), or info (request).
    "schedule_maintenance": ["book", "request"],

    # "apr": ["request", "inform"] - User asks about APR (request), system provides (inform).
    "apr": ["request", "inform"],

    # "transfer": ["request", "book"] - User asks to transfer (request), system initiates (book).
    "transfer": ["request", "book"],

    # "ingredient_substitution": ["request", "inform"] - User asks for substitution (request), system provides (inform).
    "ingredient_substitution": ["request", "inform"],

    # "calories": ["request", "inform"] - User asks for calories (request), system provides (inform).
    "calories": ["request", "inform"],

    # "current_location": ["request", "inform"] - User asks for current location (request), system provides (inform).
    "current_location": ["request", "inform"],

    # "international_fees": ["request", "inform"] - User asks about fees (request), system provides (inform).
    "international_fees": ["request", "inform"],

    # "calculator": ["request"] - User asks to calculate.
    "calculator": ["request"],

    # "definition": ["request", "inform"] - User asks for definition (request), system provides (inform).
    "definition": ["request", "inform"],

    # "next_holiday": ["request", "inform"] - User asks about next holiday (request), system provides (inform).
    "next_holiday": ["request", "inform"],

    # "update_playlist": ["request", "inform"] - User asks to update playlist (request), system confirms (inform).
    "update_playlist": ["request", "inform"],

    # "mpg": ["request", "inform"] - User asks for MPG (request), system provides (inform).
    "mpg": ["request", "inform"],

    # "min_payment": ["request", "inform"] - User asks for minimum payment (request), system provides (inform).
    "min_payment": ["request", "inform"],

    # "change_user_name": ["request", "inform"] - User asks to change username (request), system confirms (inform).
    "change_user_name": ["request", "inform"],

    # "restaurant_suggestion": ["recommend", "inform"] - User asks for suggestion (recommend), system provides (inform).
    "restaurant_suggestion": ["recommend", "inform"],

    # "travel_notification": ["request", "inform"] - User asks to set notification (request), system confirms (inform).
    "travel_notification": ["request", "inform"],

    # "cancel": ["no_book"] - User cancels booking.
    "cancel": ["no_book"],

    # "pto_used": ["inform"] - System informs user of PTO used.
    "pto_used": ["inform"],

    # "travel_suggestion": ["recommend", "inform"] - User asks for travel suggestion (recommend), system provides (inform).
    "travel_suggestion": ["recommend", "inform"],

    # "change_volume": ["request", "inform"] - User asks to change volume (request), system confirms (inform).
    "change_volume": ["request", "inform"]
}
from dataclasses import dataclass

@dataclass
class TurnMetrics:
    """Metrics for a single dialogue turn"""
    turn_id: int
    extraction_time: float
    summarization_time: float
    llm_time: float
    total_time: float
    input_tokens: int
    output_tokens: int
    context_tokens: int
    summary_length: int
    nodes_added: int
    edges_added: int
    nodes_evicted: int = 0

class ExperimentRunner:
    """Runs SSMG experiments and comparisons"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.evaluator = SSMGEvaluator()

        # Initialize LLM interface
        self.llm = GroqAPIInterface()

    def run_ssmg_experiment(self, dataset_config: Dict) -> Dict[str, Any]:
        """Run SSMG on MultiWOZ dataset"""
        logger.info("Running SSMG experiment on MultiWOZ")
    
        # Load MultiWOZ data
        start_time = time.perf_counter()
        # logger.debug("Loading MultiWOZ Dataset")
        # Make sure you have downloaded the MultiWOZ dataset manually and placed it in the specified directory.
        # For example, download from https://github.com/budzianowski/multiwoz and extract to 'data/datasets/MultiWOZ_2.2'
        data_path = dataset_config.get('data_path', 'data/datasets/MultiWOZ_2.2')
        if not Path(data_path).exists():
            logger.error(f"MultiWOZ dataset not found at {data_path}. Please download it manually from https://github.com/budzianowski/multiwoz and extract to '{data_path}'.")
            raise FileNotFoundError(f"MultiWOZ dataset not found at {data_path}. Please download it manually from https://github.com/budzianowski/multiwoz and extract to '{data_path}'.")
        multiwoz_loader = MultiWOZLoader(
            data_path=data_path,
            split='validation',
            max_dialogues=dataset_config.get('max_dialogues', 100)
        )
        total_time = time.perf_counter() - start_time
        logger.info(f"MultiWOZ Dataset Loaded in {total_time:4f} seconds")
        eval_data = multiwoz_loader.get_evaluation_data()
    
        # Initialize SSMG agent
        agent = SSMGDialogueAgent(
            llm_interface=self.llm,
            graph_config=self.config['graph'],
            summary_config=SummaryConfig(**self.config['summarizer'])
        )
    
        all_turns = []
        all_responses = []
        all_metrics = []
        ssmg_start_time = time.time()
        for dialogue in eval_data['dialogues'][:dataset_config.get('max_dialogues', 50)]:
            dialogue_id = dialogue['dialogue_id']

            # Start session for this dialogue
            dialogue_start_time = time.time()
            agent.start_session(f"ssmg_{dialogue_id}")
        
            for turn_data in dialogue['turns']:
                user_input = turn_data['user_input']
            
                try:
                    response, metrics, extracted_intent = agent.process_turn(user_input)
                
                    mapped_intent = extracted_to_true_intent.get(extracted_intent, [extracted_intent])
                    turn_record = {
                        'dialogue_id': dialogue_id,
                        'user_input': user_input,
                        'expected_response': turn_data.get('system_response', ''),
                        'generated_response': response,
                        'true_intent': turn_data.get('true_intent', 'unknown'),
                        'extraction_result': {
                            'dominant_intent': extracted_intent,
                            'mapped_intents': mapped_intent
                        },
                        'domains': dialogue['domains']
                    }
                
                    all_turns.append(turn_record)
                    all_responses.append(response)
                    all_metrics.append(metrics)
                
                except Exception as e:
                    logger.error(f"Error processing turn in {dialogue_id}: {e}")
                    continue
            dialogue_total_time = time.time() - dialogue_start_time
            logger.info(f"Dialogue {dialogue_id} completed in {dialogue_total_time:4f} seconds")
            agent.end_session()
    
        total_ssmg_time = time.time() - ssmg_start_time
        logger.info(f"SSMG Experiment completed in {total_ssmg_time:4f} seconds")
        # Evaluate results
        eval_metrics = self.evaluator.evaluate_session(all_turns, all_responses, all_metrics)
        logger.debug(f"Dialogues processed: {len(eval_data['dialogues'])} Total Turns: {len(all_turns)}")
        return {
            'method': 'ssmg',
            'eval_metrics': eval_metrics,
            'dataset_info': {
                'name': 'MultiWOZ',
                'dialogues_processed': len(eval_data['dialogues']),
                'total_turns': len(all_turns),
                'domains': eval_data['domains']
            },
            'raw_results': {
                'turns': all_turns,
                'responses': all_responses,
                'metrics': all_metrics
            }
        }


    def run_baseline_experiments(self, eval_data) -> Dict[str, Any]:
        """Run baseline methods on test dialogues"""
        baselines = {
            'full_history': FullHistoryBaseline(self.llm),
            'sliding_window_3': SlidingWindowBaseline(self.llm, window_size=3),
            'sliding_window_5': SlidingWindowBaseline(self.llm, window_size=5),
            'rag_3': RAGBaseline(self.llm, max_retrieved=3)
        }

        results = {}

        for baseline_name, baseline in baselines.items():
            logger.info(f"Running {baseline_name} baseline")

            all_turns = []
            all_responses = []
            all_metrics = []
            start_time = time.time()
            for dialogue in eval_data['dialogues']:
                dialogue_id = dialogue['dialogue_id']
            #     turns = dialogue['turns']

            #     baseline.reset()

            #     for i, turn_text in enumerate(turns):
            #         try:
            #             result = baseline.process_turn(turn_text)

            #             turn_data = {
            #                 'dialogue_id': dialogue_id,
            #                 'user_input': turn_text,
            #                 'true_intent': turn_data.get('true_intent', 'unknown'),
            #                 'extraction_result': {'dominant_intent': 'order'}
            #             }

            #             # Create mock metrics compatible with evaluation
            #             mock_metrics = type('MockMetrics', (), {
            #                 'context_tokens': result.context_tokens,
            #                 'total_time': result.latency,
            #                 'turn_id': i
            #             })()

            #             all_turns.append(turn_data)
            #             all_responses.append(result.response)
            #             all_metrics.append(mock_metrics)

            #         except Exception as e:
            #             logger.error(f"Error in {baseline_name}: {e}")
            #             continue
                for turn_data in dialogue['turns']:
                    t0 = time.perf_counter()
                    user_input = turn_data['user_input']

                    intent = turn_data.get('true_intent', 'unknown')
                
                    try:
                    #    
                        result = baseline.process_turn(user_input)
                    #    llm_time = time.perf_counter() - t0
                        # mapped_intent = extracted_to_true_intent.get(extracted_intent, [extracted_intent])
                        turn_record = {
                            'dialogue_id': dialogue_id,
                            'user_input': user_input,
                            'expected_response': turn_data.get('system_response', ''),
                            'generated_response': result.response,
                            'true_intent': intent,
                            'extraction_result': {
                                'dominant_intent': 'none',
                                'mapped_intents': intent
                            },
                            'domains': dialogue['domains']
                        }
                        finish_time = time.perf_counter() - t0
                        metrics = TurnMetrics(
                            turn_id=turn_data.get('turn_id', 0),
                            extraction_time=result.latency,
                            summarization_time=result.latency,
                            llm_time=result.latency,
                            total_time=finish_time,
                            input_tokens=result.context_tokens,
                            output_tokens=100,
                            context_tokens=result.context_tokens,
                            summary_length=0,
                            nodes_added=0,
                            edges_added=0
                        )
                    
                        all_turns.append(turn_record)
                        all_responses.append(result.response)
                        all_metrics.append(metrics)
                    
                    except Exception as e:
                        logger.error(f"Error processing turn in {dialogue_id}: {e}")
                        continue
            total_time = time.time() - start_time
            logger.info(f"{baseline_name} completed in {total_time:4f} seconds")
            eval_metrics = self.evaluator.evaluate_session(all_turns, all_responses, all_metrics)

            # results[baseline_name] = {
            #     'method': baseline_name,
            #     'eval_metrics': eval_metrics,
            #     'raw_results': {
            #         'turns': all_turns,
            #         'responses': all_responses,
            #         'metrics': all_metrics
            #     }
            # }
            results[baseline_name] = {
                'method': baseline_name,
                'eval_metrics': eval_metrics,
                'dataset_info': {
                    'name': 'MultiWOZ',
                    'dialogues_processed': len(eval_data['dialogues']),
                    'total_turns': len(all_turns),
                    'domains': eval_data['domains']
                },
                'raw_results': {
                    'turns': [],
                    'responses': [],
                    'metrics': []
                }
            }

        return results

    def run_ablation_studies(self, eval_data) -> Dict[str, Any]:
        """Run ablation studies on SSMG parameters"""
        logger.info("Running ablation studies")

        ablation_configs = [
            {'name': 'ttl_3', 'graph': {**self.config['graph'], 'max_ttl_turns': 3}},
            {'name': 'ttl_5', 'graph': {**self.config['graph'], 'max_ttl_turns': 5}},
            {'name': 'ttl_10', 'graph': {**self.config['graph'], 'max_ttl_turns': 10}},
            {'name': 'nodes_20', 'graph': {**self.config['graph'], 'max_nodes': 20}},
            {'name': 'nodes_30', 'graph': {**self.config['graph'], 'max_nodes': 30}},
            {'name': 'nodes_100', 'graph': {**self.config['graph'], 'max_nodes': 100}}
        ]

        ablation_results = {}

        for config in ablation_configs:
            logger.info(f"Running ablation: {config['name']}")

            agent = SSMGDialogueAgent(
                llm_interface=self.llm,
                graph_config=config['graph'],
                summary_config=SummaryConfig(**self.config['summarizer'])
            )

            all_turns = []
            all_responses = []
            all_metrics = []
            start_time = time.time()
            for dialogue in eval_data['dialogues']:
                dialogue_id = dialogue['dialogue_id']
                agent.start_session(f"ablation_{config['name']}_{dialogue_id}")

                for turn_data in dialogue['turns']:
                    try:
                        user_input = turn_data['user_input']
                        response, metrics, extracted_intent = agent.process_turn(user_input)
                    
                        mapped_intent = extracted_to_true_intent.get(extracted_intent, [extracted_intent])
                        turn_record = {
                            'dialogue_id': dialogue_id,
                            'user_input': user_input,
                            'expected_response': turn_data.get('system_response', ''),
                            'generated_response': response,
                            'true_intent': turn_data.get('true_intent', 'unknown'),
                            'extraction_result': {
                                'dominant_intent': extracted_intent,
                                'mapped_intents': mapped_intent
                            },
                            'domains': dialogue['domains']
                        }
                    
                        all_turns.append(turn_record)
                        all_responses.append(response)
                        all_metrics.append(metrics)

                    except Exception as e:
                        logger.error(f"Ablation error: {e}")
                        continue

                agent.end_session()
            total_time = time.time() - start_time
            logger.info(f"Ablation {config['name']} completed in {total_time:4f} seconds")
            eval_metrics = self.evaluator.evaluate_session(all_turns, all_responses, all_metrics)
            ablation_results[config['name']] = {
                'method': config['name'],
                'eval_metrics': eval_metrics,
                'dataset_info': {
                    'name': 'MultiWOZ',
                    'dialogues_processed': len(eval_data['dialogues']),
                    'total_turns': len(all_turns),
                    'domains': eval_data['domains']
                },
                'raw_results': {
                    'turns': [],
                    'responses': [],
                    'metrics': []
                }
            }

        return ablation_results

    def generate_plots(self, results: Dict[str, Any], output_dir: str):
        """Generate evaluation plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Prepare data for plotting
        methods = []
        accuracies = []
        latencies = []
        token_usages = []
        constraint_adherences = []
        reference_resolutions = []
        token_efficiencies = []
        latency_efficiencies = []
        avg_response_lengths = []
        error_task_failures = []
        error_constraint_violations = []
        error_reference_failures = []

        for method, result in results.items():
            if 'eval_metrics' in result:
                metrics = result['eval_metrics']
                methods.append(method)
                accuracies.append(metrics.get('turn_accuracy', 0.0))
                latencies.append(metrics.get('avg_latency', 0.0))
                token_usages.append(metrics.get('avg_context_tokens', 0.0))
                constraint_adherences.append(metrics.get('constraint_adherence', 0.0))
                reference_resolutions.append(metrics.get('reference_resolution_accuracy', 0.0))
                token_efficiencies.append(metrics.get('success_per_thousand_tokens', 0.0))
                latency_efficiencies.append(metrics.get('success_per_second_latency', 0.0))
                avg_response_lengths.append(metrics.get('average_response_length', 0.0))
                error_analysis = metrics.get('error_analysis', {})
                error_task_failures.append(error_analysis.get('task_failures', 0))
                error_constraint_violations.append(error_analysis.get('constraint_violations', 0))
                error_reference_failures.append(error_analysis.get('reference_resolution_failures', 0))

        # Create comparison plots
        fig, axes = plt.subplots(3, 3, figsize=(22, 16))

        # Accuracy comparison
        axes[0, 0].bar(methods, accuracies)
        axes[0, 0].set_title('Turn Accuracy by Method')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Latency comparison
        axes[0, 1].bar(methods, latencies)
        axes[0, 1].set_title('Average Latency by Method')
        axes[0, 1].set_ylabel('Latency (s)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Token usage comparison
        axes[0, 2].bar(methods, token_usages)
        axes[0, 2].set_title('Average Context Tokens by Method')
        axes[0, 2].set_ylabel('Tokens')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Constraint adherence
        axes[1, 0].bar(methods, constraint_adherences)
        axes[1, 0].set_title('Constraint Adherence by Method')
        axes[1, 0].set_ylabel('Adherence Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Reference resolution accuracy
        axes[1, 1].bar(methods, reference_resolutions)
        axes[1, 1].set_title('Reference Resolution Accuracy by Method')
        axes[1, 1].set_ylabel('Resolution Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Average response length
        axes[1, 2].bar(methods, avg_response_lengths)
        axes[1, 2].set_title('Average Response Length by Method')
        axes[1, 2].set_ylabel('Words')
        axes[1, 2].tick_params(axis='x', rotation=45)

        # Efficiency scatter plot (token efficiency)
        axes[2, 0].scatter(token_usages, token_efficiencies)
        for i, method in enumerate(methods):
            axes[2, 0].annotate(method, (token_usages[i], token_efficiencies[i]))
        axes[2, 0].set_xlabel('Context Tokens')
        axes[2, 0].set_ylabel('Token Efficiency')
        axes[2, 0].set_title('Success per Thousand Tokens vs Context Tokens')

        # Efficiency scatter plot (latency efficiency)
        axes[2, 1].scatter(latencies, latency_efficiencies)
        for i, method in enumerate(methods):
            axes[2, 1].annotate(method, (latencies[i], latency_efficiencies[i]))
        axes[2, 1].set_xlabel('Latency (s)')
        axes[2, 1].set_ylabel('Latency Efficiency')
        axes[2, 1].set_title('Success per Second Latency vs Latency')

        # Error analysis bar plot
        width = 0.25
        x = range(len(methods))
        axes[2, 2].bar([i - width for i in x], error_task_failures, width=width, label='Task Failures')
        axes[2, 2].bar(x, error_constraint_violations, width=width, label='Constraint Violations')
        axes[2, 2].bar([i + width for i in x], error_reference_failures, width=width, label='Reference Failures')
        axes[2, 2].set_xticks(x)
        axes[2, 2].set_xticklabels(methods, rotation=45)
        axes[2, 2].set_title('Error Analysis by Method')
        axes[2, 2].set_ylabel('Count')
        axes[2, 2].legend()

        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path_dir = output_path / f'comparison_plots_{timestamp}.png'
        plt.savefig(output_path_dir, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Plots saved to {output_path_dir}")

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file"""

        # Convert evaluation metrics to serializable format
        serializable_results = {}
        for method, result in results.items():
            if 'eval_metrics' in result:
                metrics = result['eval_metrics']
                serializable_results[method] = metrics
            else:
                serializable_results[method] = result

        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        except:
            import yaml
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'multiwoz_results_{timestamp}.yaml'
            # output_dir.mkdir(exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(serializable_results, f)

        logger.info(f"Results saved to {output_file}")

class SyntheticDialogueTester:
    """Run SSMG on synthetic dialogues to test model performance"""

    def __init__(self, config: Dict[str, Any], llm_interface=None):
        self.config = config
        self.llm = llm_interface if llm_interface else GroqAPIInterface()
        self.agent = SSMGDialogueAgent(
            llm_interface=self.llm,
            graph_config=config['graph'],
            summary_config=SummaryConfig(**config['summarizer'])
        )

    def run_on_dialogues(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run SSMG on a list of synthetic dialogues
        Returns results with user_input, response, and metrics
        """
        all_turns = []
        all_responses = []
        all_metrics = []

        start_time = time.time()
        for dialogue in dialogues:
            dialogue_id = dialogue.get('dialogue_id', 'unknown')
            self.agent.start_session(f"synthetic_{dialogue_id}")

            for turn in dialogue['turns']:
                user_input = turn['user_input']
                try:
                    response, metrics, extracted_intnet = self.agent.process_turn(user_input)
                    turn_record = {
                        'dialogue_id': dialogue_id,
                        'user_input': user_input,
                        'expected_response': turn.get('system_response', ''),
                        'response': response,
                        'true_intent': turn.get('true_intent', 'unknown'),
                        'extraction_result': {'dominant_intent': extracted_intnet},
                        'metrics': metrics
                    }
                    all_turns.append(turn_record)
                    all_responses.append(response)
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Error processing turn in {dialogue_id}: {e}")
                    continue

            self.agent.end_session()
        total_time = time.time() - start_time
        logger.info(f"Processed {len(dialogues)} synthetic dialogues in {total_time:.2f}s")

        return {
            'total_dialogues': len(dialogues),
            'total_turns': len(all_turns),
            'turns': all_turns,
            'responses': all_responses,
            'metrics': all_metrics,
            'total_time': total_time
        }

    def print_summary(self, results: Dict[str, Any]):
        """Print a quick summary of synthetic dialogue results"""
        print("\n🎯 Synthetic Dialogue Test Summary")
        print(f"Total Dialogues: {results['total_dialogues']}")
        print(f"Total Turns: {results['total_turns']}")
        print(f"Total Time: {results['total_time']:.2f}s")
        print("\nSample Responses:")
        for turn in results['turns'][:5]:  # print first 5 turns
            print(f"[{turn['dialogue_id']}] User: {turn['user_input']}")
            print(f"             Expected: {turn['expected_response']}")
            print(f"             Response: {turn['response']}\n")


def main():
    parser = argparse.ArgumentParser(description='Run SSMG evaluation on MultiWOZ')
    parser.add_argument('--config', default='configs/default_config.json')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--max-dialogues', type=int, default=100, 
                       help='Maximum number of dialogues to evaluate')
    parser.add_argument('--run-baselines', action='store_true', help='Run baseline experiments')
    parser.add_argument('--run-ablations', action='store_true', help='Run ablation studies')
    parser.add_argument('--run-ssmg', action='store_true', default=False, help='Run SSMG experiment (default: False)')

    args = parser.parse_args()

    # Load configuration
    config = ConfigLoader.load_config(args.config)

    # Update config with MultiWOZ settings
    config['evaluation']['max_dialogues'] = args.max_dialogues

    # Create output directory  
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize runner
    runner = ExperimentRunner(config)

    # Run experiments
    results = {}

    # Run SSMG on MultiWOZ if requested
    if args.run_ssmg:
        ssmg_results = runner.run_ssmg_experiment(config['evaluation'])
        results['ssmg'] = ssmg_results

    # Run baselines if requested
    if args.run_baselines:
        # For baselines, need to load test dialogues
        # Use the same loader as SSMG experiment for consistency
        data_path = config['evaluation'].get('data_path', 'data/datasets')
        if not Path(data_path).exists():
            logger.error(f"MultiWOZ dataset not found at {data_path}. Please download it manually from https://github.com/budzianowski/multiwoz and extract to '{data_path}'.")
            raise FileNotFoundError(f"MultiWOZ dataset not found at {data_path}. Please download it manually from https://github.com/budzianowski/multiwoz and extract to '{data_path}'.")
        multiwoz_loader = MultiWOZLoader(
            data_path=data_path,
            split='validation',
            max_dialogues=config['evaluation'].get('max_dialogues', 100)
        )
        eval_data = multiwoz_loader.get_evaluation_data()
        # test_dialogues = eval_data['dialogues']
        baseline_results = runner.run_baseline_experiments(eval_data)
        results['baseline'] = baseline_results

    # Run ablation studies if requested
    if args.run_ablations:
        data_path = config['evaluation'].get('data_path', 'data/datasets')
        if not Path(data_path).exists():
            logger.error(f"MultiWOZ dataset not found at {data_path}. Please download it manually from https://github.com/budzianowski/multiwoz and extract to '{data_path}'.")
            raise FileNotFoundError(f"MultiWOZ dataset not found at {data_path}. Please download it manually from https://github.com/budzianowski/multiwoz and extract to '{data_path}'.")
        multiwoz_loader = MultiWOZLoader(
            data_path=data_path,
            split='validation',
            max_dialogues=config['evaluation'].get('max_dialogues', 100)
        )
        eval_data = multiwoz_loader.get_evaluation_data()
        # test_dialogues = eval_data['dialogues']
        ablation_results = runner.run_ablation_studies(eval_data)
        results['ablations'] = ablation_results

    # Save and visualize results if any
    if results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'multiwoz_results_{timestamp}.json'
        runner.save_results(results, str(output_file))
        runner.generate_plots(results, str(output_dir))

        # Print summary
        print("\n🎯 MultiWOZ Evaluation Results:")
        for method, result in results.items():
            if 'eval_metrics' in result:
                metrics = result['eval_metrics']
                print(f"{method}: Accuracy={metrics.get('turn_accuracy', 0.0):.3f}, "
                      f"Tokens/Turn={metrics.get('avg_context_tokens', 0.0):.1f}, "
                      f"Latency={metrics.get('avg_latency', 0.0):.3f}s")
            


if __name__ == '__main__':
    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='evaluation_baseline.log',  
    filemode='w'  
    )
    logger = logging.getLogger(__name__)
    main()


# python -m evaluation.eval_runner --config configs/default_config.json --output-dir results --max-dialogues 100 --run-ssmg
# python -m evaluation.eval_runner --config configs/default_config.json --output-dir results --max-dialogues 100 --run-baselines

