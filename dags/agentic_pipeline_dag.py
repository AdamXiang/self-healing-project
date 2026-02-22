import itertools
import json
import re
from datetime import timedelta, datetime

from airflow.sdk import dag, task, Param, get_current_context
import logging
import os

from sympy.physics.units import temperature

logger = logging.getLogger(__name__)


class Config:
    BASE_DIR = os.getenv('PIPELINE_BASE_DIR', '/Users/adam/Desktop/project/self-healing')
    INPUT_FILE = os.getenv('PIPELINE_INPUT_FILE', f'{BASE_DIR}/input/academic_dataset_review.json')
    OUTPUT_DIR = os.getenv('PIPELINE_OUTPUT_DIR', f'{BASE_DIR}/output')


    MAX_TEXT_LENGTH = int(os.getenv('PIPELINE_MAX_TEXT_LENGTH', 2000))
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_OFFSET = 0


    # Ollama settings
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 120))
    OLLAMA_RETRIES = int(os.getenv('OLLAMA_RETRIES', 3))


default_args = {
    'owner': 'adam',
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
    'execution_timeout': timedelta(minutes=30),
}



def _load_ollama_model(model_name: str):
    import ollama

    logger.info(f'Loading OLLAMA Model: {model_name}')
    logger.info(f'OLLAMA Host: {Config.OLLAMA_HOST}')

    client = ollama.Client(host=Config.OLLAMA_HOST)

    try:
        client.show(model_name)

        logger.info(f'OLLAMA Model: {model_name} is available.')

    except ollama.RequestError as e:
        # if error occurs, we pull the model_name
        logger.info('OLLAMA Model not found locally. Attempting to pull from remote repository...')

        try:
            client.pull(model_name)
            logger.info(f'OLLAMA Model: {model_name} pulled successfully.')

        except ollama.RequestError as e:
            logger.error(f'OLLAMA Model: {model_name} could not be pulled: {e}')
            raise

    test_response = client.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Classify the sentiment: 'This is a great product!' as positive, negative, or neutral."
            }
        ]
    )

    test_result = test_response['message']['content'].strip().upper()
    logger.info(f'OLLAMA Model: {model_name} test result: {test_result}')

    return {
        'backend': 'ollama',
        'model_name': model_name,
        'ollama_host': Config.OLLAMA_HOST,
        'max_length': Config.MAX_TEXT_LENGTH,
        'status': 'loaded',
        'validated_at': datetime.now().isoformat()
    }


def _load_from_file(params: dict, batch_size: int, offset: int):
    input_file = params.get('input_file', Config.INPUT_FILE)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f'Input file {input_file} not found.')

    reviews = []

    with open(input_file, 'r', encoding='utf-8') as f:
        sliced = itertools.islice(f, offset, offset + batch_size)

        for line in sliced:
            try:
                review = json.loads(line.strip())
                reviews.append({
                    'review_id': review.get('review_id'),
                    'user_id': review.get('user_id'),
                    'business_id': review.get('business_id'),
                    'stars': review.get('stars', 0),
                    'text': review.get('text'),
                    'date': review.get('date'),
                    'useful': review.get('useful', 0),
                    'funny': review.get('funny', 0),
                    'cool': review.get('cool', 0)
                })
            except json.decoder.JSONDecodeError as e:
                logger.warning(f'JSON Decode Error: {e}')
                continue

    logger.info(f'Read {len(reviews)} reviews from {input_file}.')

    return reviews


def _parse_ollama_response(response_text: str):
    try:
        clean_text = response_text.strip()

        if clean_text.startswith('```'):
            # deal with markdown format
            # if our text starts with ``` then checks whether has three backticks in the end
            lines = clean_text.split('\n')
            clean_text = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])

        parsed = json.loads(clean_text)
        sentiment = parsed.get('sentiment', 'NEUTRAL').upper()
        confidence = float(parsed.get('confidence', 0.0))


        if sentiment not in ['NEUTRAL', 'POSITIVE', 'NEGATIVE']:
            sentiment = 'NEUTRAL'

        return {
            'label': sentiment,
            'score': min(max(confidence, 0.0), 1.0),
        }

    except (json.decoder.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        upper_text = response_text.strip().upper()

        if 'POSITIVE' in upper_text:
            return {
                'label': 'POSITIVE',
                'score': 0.75
            }
        elif 'NEGATIVE' in upper_text:
            return {
                'label': 'NEGATIVE',
                'score': 0.75
            }
        return {
            'label': 'NEUTRAL',
            'score': 0.5
        }

def _heal_review(review: dict):
    text = review.get('text', '')

    result = {
        'review_id': review.get('review_id'),
        'business_id': review.get('business_id'),
        'stars': review.get('stars', 0),
        'original_text': None,
        'error_type': None,
        'action_taken': None,
        'was_healed': False,
        'metadata': {
            'user_id': review.get('user_id'),
            'date': review.get('date'),
            'useful': review.get('useful', 0),
            'funny': review.get('funny', 0),
            'cool': review.get('cool', 0),
        },
    }

    if isinstance(text, (str, float, int, type(None))):
        result['original_text'] = text
    else:
        result['original_text'] = str(text) if text else None

    if text is None:
        result['error_type'] = 'missing_text'
        result['action_taken'] = 'filled_with_placeholder'
        result['was_healed'] = True
        result['healed_text'] = 'No review text provided.'
        return result
    elif not isinstance(text, str):
        result['error_type'] = 'wrong_type'
        try:
            converted = str(text).strip()
            result['healed_text'] = converted if converted else 'No review text provided.'
        except Exception as e:
            result['healed_text'] = 'Conversion failed.'

        result['action_taken'] = 'type_conversion'
        result['was_healed'] = True
    elif not text.strip():
        result['error_type'] = 'empty_text'
        result['action_taken'] = 'filled_with_placeholder'
        result['was_healed'] = True
        result['healed_text'] = 'No review text provided.'
    elif not re.search(r'[a-zA-Z0-9]', text):
        result['error_type'] = 'special_characters_only'
        result['action_taken'] = 'replaced_special_characters_only'
        result['was_healed'] = True
        result['healed_text'] = 'Non-text content'
    elif len(text) > Config.MAX_TEXT_LENGTH:
        result['error_type'] = 'too_long'
        result['action_taken'] = 'truncated_text'
        result['was_healed'] = True
        result['healed_text'] = text[:Config.MAX_TEXT_LENGTH - 3] + '...'
    else:
        result['healed_text'] = text.strip()
        result['was_healed'] = False

    return result


def _analyze_with_ollama(healed_reviews: list[dict], model_info: dict):
    import ollama
    import time

    model_name = model_info.get('model_name')
    ollama_host = model_info.get('ollama_host', Config.OLLAMA_HOST)

    try:
        client = ollama.Client(ollama_host)
    except Exception as e:
        logging.error(f'Failed to connect to {ollama_host}: {e}')
        return _created_degraded_results(healed_reviews, str(e))

    results = []
    total = len(healed_reviews)


    for idx, review in enumerate(healed_reviews):
        text = review.get('healed_text', '')
        prediction = None

        for attempt in range(Config.OLLAMA_RETRIES):
            try:

                prompt = (
                    f"""
                    Analyze the sentiment of this reviews and classify it as POSITIVE, NEGATIVE, NEUTRAL. 
                    
                    Review: '{text}'
                    
                    Reply with ONLY a JSON object: {{'sentiment': 'POSITIVE, 'confidence': 0.95}}.
                    """
                )

                response = client.chat(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.1}
                )

                response_text = response['messages']['content'].strip()
                prediction = _parse_ollama_response(response_text)
                break

            except Exception as e:
                if attempt < Config.OLLAMA_RETRIES - 1:
                    logger.warning(f'Attempt {attempt + 1} failed for reviews {review.get('review_id')}: {e}. Retrying...')
                    time.sleep(1)
                else:
                    logger.error(f'All attempts failed for review {review.get('review_id')}: {e}.')
                    prediction = {'label': 'NEUTRAL', 'score': 0.5, 'error': str(e)}

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            logger.info(f'{idx + 1}/{total} reviews analyzed.')


        results.append(
            {
            'review_id': review.get('review_id'),
            'business_id': review.get('business_id'),
            'stars': review.get('stars', 0),
            'text': review.get('healed_text', ''),
            'original_text': review.get('original_text', ''),
            'predicted_sentiment': prediction.get('label'),
            'confidence': round(prediction.get('score'), 4),
            'status': 'healed' if review.get('was_healed') else 'success',
            'healing_applied': review.get('was_healed'),
            'healing_action': review.get('action_taken') if review.get('was_healed') else None,
            'error_type': review.get('error_type') if review.get('was_healed') else None,
            'metadata': review.get('metadata', {})
            }
        )

    logger.info(f'Ollama inference complete: {len(results)} reviews analyzed.')
    return results


def _created_degraded_results(healed_reviews: list[dict], error_message: str):
    return [
        {
            **review,
            'text': review.get('headed_text', ''),
            'prediction_sentiment': 'NEUTRAL',
            'confidence': 0.5,
            'status': 'degraded',
            'error_message': error_message,
        }
        for review in healed_reviews
    ]

@dag(
    dag_id='self_healing_pipeline_dag',
    default_args=default_args,
    description='Pipeline for sentiment analysis using OLLAMA',
    tags=['self-healing', 'ollama', 'nlp', 'reviews', 'sentiment_analysis'],
    schedule=None,
    start_date=datetime(2026, 2, 22),
    catchup=False,
    params={
        'input_file': Param(
            default=Config.INPUT_FILE,
            type='string',
            description='Path to the input JSON file containing reviews',
        ),
        'output_dir': Param(
            default=Config.OUTPUT_DIR,
            type='string',
            description='Directory to write output files to',
        ),
        'batch_size': Param(
            default=Config.DEFAULT_BATCH_SIZE,
            type='integer',
            description='Number of reviews to process in each batch',
        ),
        'offset': Param(
            default=Config.DEFAULT_OFFSET,
            type='integer',
            description='Offset to start reading reviews from the input file',
        ),
        'ollama_model': Param(
            default=Config.OLLAMA_MODEL,
            type='string',
            description='Name of the OLLAMA model to use for sentiment analysis'
        )
    },
    render_template_as_native_obj=True,
)
def self_healing_pipeline_dag(**kwargs):
    @task
    def load_model():
        context = get_current_context()
        params = context['params']
        model_name = params.get('ollama_model', Config.OLLAMA_MODEL)
        logger.info(f'Loading OLLAMA Model: {model_name}')
        return _load_ollama_model(model_name)

    @task
    def load_reviews():
        context = get_current_context()
        params = context['params']
        batch_size = params.get('batch_size', Config.DEFAULT_BATCH_SIZE)
        offset = params.get('offset', Config.DEFAULT_OFFSET)
        logger.info(f'Loading OLLAMA Reviews: {offset}/{batch_size}')
        return _load_from_file(params, batch_size, offset)

    @task
    def diagnose_and_heal_batch(reviews: list[dict]):
        healed_reviews = [ _heal_review(review) for review in reviews ]
        healed_count = sum(1 for r in healed_reviews if r.get('was_healed', False))
        logger.info(f'Healed {healed_count} reviews')

        return healed_reviews

    @task
    def batch_analyze_sentiment(healed_reviews: list[dict], model_info: dict):
        if not healed_reviews:
            logger.info(f'No reviews to analyze')
            return []

        logger.info(f'Analyzing {len(healed_reviews)} reviews')
        return _analyze_with_ollama(healed_reviews, model_info)


    @task
    def aggregate_results(results: list[list[dict]]):
        context = get_current_context()
        params = context['params']
        results = list(results)

        total = len(results)

        success_count = sum(1 for r in results if r.get('status') == 'success')
        healed_count = sum(1 for r in results if r.get('status') == 'healed')
        degraded_count = sum(1 for r in results if r.get('status') == 'degraded')

        sentiment_dist = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}

        for r in results:
            sentiment = r.get('prediction_sentiment', 'NEUTRAL')
            sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1


        healing_stats = {}

        for r in results:
            if r.get('healed_applied'):
                action = r.get('healing_action', 'UNKNOWN')
                healing_stats[action] = healing_stats.get(action, 0) + 1

        star_sentiment_stats = {}
        for r in results:
            stars = r.get('stars', 0)
            sentiment = r.get('prediction_sentiment', 'NEUTRAL')
            if stars and sentiment:
                key = f'{int(stars)}_star'

                if key not in star_sentiment_stats:
                    star_sentiment_stats[stars] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}

            star_sentiment_stats[stars][sentiment] += 1

        # expected output like {'1_star': {'POSITIVE': 10, 'NEGATIVE': 50, 'NEUTRAL': 23}, .....}

        confidence_stats = {'success': [], 'healed': [], 'degraded': []}
        for r in results:
            status = r.get('status')
            confidence = r.get('confidence', 0.0)

            if status in confidence_stats:
                confidence_stats[status].append(confidence)

        avg_confidence = {
            status: (sum(conf_list) / len(conf_list)) if conf_list else 0
                     for status, conf_list in confidence_stats.items()
        }

        summary = {
            'run_info': {
                'timestamp': datetime.now().isoformat(),
                'batch_size': params.get('batch_size', Config.DEFAULT_BATCH_SIZE),
                'offset': params.get('offset', Config.DEFAULT_OFFSET),
                'input_file': params.get('input_file', Config.INPUT_FILE),
            },
            'totals': {
                'processed': total,
                'success': success_count,
                'healed': healed_count,
                'degraded': degraded_count,
            },
            'rates': {
                'success_rate': round(success_count / max(total, 1), 4),
                'healed_rate': round(healed_count / max(total, 1), 4),
                'degraded_rate': round(degraded_count / max(total, 1), 4),
            },
            'sentiment_distribution': sentiment_dist,
            'healing_statistics': healing_stats,
            'star_sentiment_correlation': star_sentiment_stats,
            'average_confidence': avg_confidence,
            'results': results,
        }

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        offset = params.get('offset', Config.DEFAULT_OFFSET)
        output_file = f'{Config.OUTPUT_DIR}/sentiment_analysis_summary_{timestamp}_{offset}.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4, default=str)

        logger.info(f'Saving results to {output_file}')
        logger.info(f'Processing results: {len(results)}')

        return {
            k: v for k, v in summary.items() if k != 'results'
        }

    @task
    def generate_health_report(summary: dict):
        total = summary['totals']['processed']
        success = summary['totals']['success']
        healed = summary['totals']['healed']
        degraded = summary['totals']['degraded']

        if degraded > total * 0.1:
            health_status = 'CRITICAL'
        elif degraded > 0:
            health_status = 'DEGRADED'
        elif healed > total * 0.5:
            health_status = 'WARNING'
        else:
            health_status = 'HEALTHY'

        report = {
            'pipeline': 'self_healing_pipeline',
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'run_info': summary['run_info'],
            'metrics': {
                'total_processed': total,
                'success_count': success,
                'healed_count': healed,
                'degraded_count': degraded,
                'success_rate': summary['rates']['success_rate'],
                'healed_rate': summary['rates']['healed_rate'],
                'degraded_rate': summary['rates']['degraded_rate'],
            },
            'sentiment_distribution': summary['sentiment_distribution'],
            'healing_statistics': summary['healing_statistics'],
            'average_confidence': summary['average_confidence'],
        }

        logger.info(f'Healing pipeline results: {json.dumps(report, indent=2)}')
        return report

    model_info = load_model()
    reviews = load_reviews()

    healed_reviews = diagnose_and_heal_batch(reviews)
    analyzed_results = batch_analyze_sentiment(healed_reviews, model_info)

    summary = aggregate_results(analyzed_results)
    health_report = generate_health_report(summary)


self_healing_pipeline = self_healing_pipeline_dag()