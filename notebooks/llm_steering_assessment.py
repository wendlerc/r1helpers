# %%
# Imports and Configuration
import json
import os
from openai import OpenAI
from typing import List, Dict, Any
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configuration - List of features to process
FEATURES = [
    {
        "feature": 188,
        "direction": "uncertainty or confusion",
        "file_path": "/disk/u/troitskiid/projects/r1helpers/results/llama8b_r1_math500_steering_layer15_feature188_strength1.5_steer100_seed42_summary.jsonl"
    },
    {
        "feature": 744,
        "direction": "going to initial approach, starting from the beginning",
        "file_path": "/disk/u/troitskiid/projects/r1helpers/results/llama8b_r1_math500_steering_layer15_feature744_strength1.5_steer100_seed42_summary.jsonl"
    },
    {
        "feature": 31748,
        "direction": "knowledge recall, fact recall - sometimes random or unrelated knowledge",
        "file_path": "/disk/u/troitskiid/projects/r1helpers/results/llama8b_r1_math500_steering_layer15_feature31748_strength1.5_steer100_seed42_summary.jsonl"
    },
    {
        "feature": 25929,
        "direction": "coming to conclusion, wrapping things up",
        "file_path": "/disk/u/troitskiid/projects/r1helpers/results/llama8b_r1_math500_steering_layer15_feature25929_strength1.5_steer100_seed42_summary.jsonl"
    }
]

API_MODEL = "openai/gpt-5-nano"  
MAX_EXAMPLES = None
MAX_TOKENS = 5000

# Parallelism
BATCH_SIZE = 100            # number of examples per parallel batch
MAX_WORKERS = BATCH_SIZE   # number of concurrent API calls

# Initialize OpenRouter client
# Make sure to set your OPENROUTER_API_KEY environment variable
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# %%
# Assessment Functions

# Assessment Functions

def clean_json_response(json_str: str) -> str:
    """
    Clean common JSON formatting issues from LLM responses.
    """
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Remove trailing commas at the end of the string (before potential whitespace)
    json_str = re.sub(r',\s*$', '', json_str)
    
    # Fix invalid escape sequences in string values
    # First, let's handle strings with invalid escapes by finding string content
    def fix_string_escapes(match):
        string_content = match.group(1)
        # Replace single backslashes with double backslashes for valid JSON escapes
        # Handle common invalid escapes that LLMs generate
        string_content = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', string_content)  # Escape invalid backslashes
        return f'"{string_content}"'
    
    # Apply escape fixing to string literals (content between quotes)
    json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', fix_string_escapes, json_str)
    
    # Remove any extra whitespace
    json_str = json_str.strip()
    
    return json_str

def create_assessment_prompt(text_before: str, original_text_after: str, steered_text_after: str, steering_direction: str) -> str:
    """
    Create a prompt for the AI to assess how well the example matches the steering direction.
    """
    prompt = f"""You are an expert evaluator of language model behavior and steering interventions.

Please analyze the following example and assess how much the steered response corresponds to the steering direction "{steering_direction}" on a scale from 1 to 100, where:

- 1-20: Very little to no correspondence with the steering direction
- 21-40: Some correspondence but not strong
- 41-60: Moderate correspondence
- 61-80: Strong correspondence
- 81-100: Very strong/excellent correspondence with the steering direction

**Original Context:**
{text_before}

**Original Response (before steering):**
{original_text_after}

**Steered Response (after steering):**
{steered_text_after}

**Assessment Criteria:**
- Consider how well the steered response reflects the intended steering direction "{steering_direction}"
- Look for behavioral changes, linguistic patterns, and content shifts
- Compare the steered response to the original response
- Evaluate the consistency and strength of the steering effect

Please provide your assessment as a JSON object with the following format:
{{
    "score": <number between 1-100>,
    "reasoning": "<brief explanation of your assessment>",
}}

Assessment:"""

    return prompt

def assess_example(text_before: str, original_text_after: str, steered_text_after: str, steering_direction: str) -> Dict[str, Any]:
    """
    Send an assessment request to OpenRouter API and return the result.
    """
    prompt = create_assessment_prompt(text_before, original_text_after, steered_text_after, steering_direction)

    try:
        completion = client.chat.completions.create(
            model=API_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # temperature=0.1,  # Low temperature for consistent assessments
            max_tokens=MAX_TOKENS
        )

        response_text = completion.choices[0].message.content

        # Try to parse the JSON response
        try:
            # First, handle reasoning traces with <think> tags
            json_content = response_text.strip()
            
            # If there's a </think> tag, extract everything after it
            if "</think>" in json_content:
                think_end = json_content.find("</think>") + 8  # +8 to include the closing tag
                json_content = json_content[think_end:].strip()
            
            # Then handle markdown code blocks
            if "```json" in json_content:
                json_start = json_content.find("```json") + 7
                json_end = json_content.find("```", json_start)
                if json_end == -1:  # If no closing ``` found
                    json_end = len(json_content)
                json_content = json_content[json_start:json_end].strip()
            elif "```" in json_content:
                json_start = json_content.find("```") + 3
                json_end = json_content.find("```", json_start)
                if json_end == -1:  # If no closing ``` found
                    json_end = len(json_content)
                json_content = json_content[json_start:json_end].strip()

            # Clean common JSON formatting issues
            json_content = clean_json_response(json_content)

            assessment = json.loads(json_content)

            # Validate the assessment structure
            if not isinstance(assessment.get("score"), (int, float)):
                raise ValueError("Score must be a number")
            if not (1 <= assessment["score"] <= 100):
                raise ValueError("Score must be between 1 and 100")

            return assessment

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing assessment response: {e}")
            print(f"Raw response: {response_text}")
            # Return a default assessment
            return {
                "score": 50,
                "reasoning": "Error parsing AI response",
                "key_changes": "Unable to analyze"
            }

    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        return {
            "score": 50,
            "reasoning": f"API Error: {str(e)}",
            "key_changes": "Unable to analyze"
        }

# %%
# File Processing Functions

def save_assessments(assessments: List[Dict], output_file: str):
    """
    Save assessments to JSONL file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for assessment in assessments:
            f.write(json.dumps(assessment) + '\n')

def process_jsonl_file(file_path: str, steering_direction: str, output_file: str, max_examples: int = None):
    """
    Process the JSONL file and assess each example in parallel batches.
    """
    assessments = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_lines = len(lines)
    num_to_process = min(max_examples, num_lines) if max_examples is not None else num_lines
    total_examples = num_to_process
    print(f"Processing {total_examples} examples...")

    for batch_start in range(0, num_to_process, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_to_process)

        # Prepare batch
        batch_items = []
        for i in range(batch_start, batch_end):
            try:
                entry = json.loads(lines[i].strip())

                text_before = entry.get('text_before', '')
                original_text_after = entry.get('original_text_after', '')
                steered_text_after = entry.get('steered_text_after', '')
                unique_id = entry.get('unique_id', f'example_{i}')

                if not all([text_before, original_text_after, steered_text_after]):
                    print(f"Skipping example {i}: Missing required fields")
                    continue

                batch_items.append((i, unique_id, text_before, original_text_after, steered_text_after))
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue

        if not batch_items:
            continue

        print(f"Assessing examples {batch_start + 1}-{batch_start + len(batch_items)}/{total_examples}")

        # Parallel API calls for the batch
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_meta = {
                executor.submit(
                    assess_example, tb, oa, sa, steering_direction
                ): (i, unique_id, tb, oa, sa)
                for (i, unique_id, tb, oa, sa) in batch_items
            }

            for future in as_completed(future_to_meta):
                i, unique_id, tb, oa, sa = future_to_meta[future]
                try:
                    assessment = future.result()
                except Exception as e:
                    print(f"Error processing example {i}: {e}")
                    assessment = {
                        "score": 50,
                        "reasoning": f"Thread error: {str(e)}",
                        "key_changes": "Unable to analyze"
                    }

                result = {
                    "unique_id": unique_id,
                    "steering_direction": steering_direction,
                    "assessment": assessment,
                    "original_fields": {
                        "text_before": tb[:200] + "..." if len(tb) > 200 else tb,
                        "original_text_after": oa[:200] + "..." if len(oa) > 200 else oa,
                        "steered_text_after": sa[:200] + "..." if len(sa) > 200 else sa
                    }
                }
                assessments.append(result)

        # Save progress after each batch
        save_assessments(assessments, output_file)
        print(f"Saved progress after {len(assessments)} examples")

    # Final save
    save_assessments(assessments, output_file)
    print(f"Completed assessment of {len(assessments)} examples")
    print(f"Results saved to: {output_file}")

    return assessments

# %%
# Main Processing and Analysis

def main():
    """
    Main function to run the assessment script.
    """
    print("Starting steering assessment script for multiple features...")
    print(f"Model: {API_MODEL}")
    print(f"Batch size: {BATCH_SIZE}, Max workers: {MAX_WORKERS}")
    print(f"Features to process: {len(FEATURES)}")

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: Please set the OPENROUTER_API_KEY environment variable")
        return

    # Process each feature
    all_results = {}
    
    for feature_config in FEATURES:
        feature = feature_config["feature"]
        steering_direction = feature_config["direction"]
        jsonl_file_path = feature_config["file_path"]
        
        # Generate output file path
        output_file = jsonl_file_path.replace("summary.jsonl", "steering_assessment.jsonl").replace("/results/", "/results/llm_steering_assessments/")
        
        print(f"\n{'='*60}")
        print(f"Processing Feature {feature}")
        print(f"{'='*60}")
        print(f"Input file: {jsonl_file_path}")
        print(f"Steering direction: {steering_direction}")
        print(f"Output file: {output_file}")

        # Check if input file exists
        if not os.path.exists(jsonl_file_path):
            print(f"Error: Input file {jsonl_file_path} does not exist")
            continue

        # Process the file
        assessments = process_jsonl_file(
            jsonl_file_path,
            steering_direction,
            output_file,
            max_examples=MAX_EXAMPLES
        )

        # Store results for this feature
        all_results[feature] = {
            "direction": steering_direction,
            "output_file": output_file,
            "assessments": assessments
        }

        # Print summary statistics for this feature
        if assessments:
            scores = [a["assessment"]["score"] for a in assessments if isinstance(a["assessment"].get("score"), (int, float))]
            if scores:
                avg_score = sum(scores) / len(scores)
                median_score = statistics.median(scores)
                min_score = min(scores)
                max_score = max(scores)
                print(f"\nFeature {feature} Summary Statistics:")
                print(f"Average score: {avg_score:.2f}")
                print(f"Median score: {median_score:.2f}")
                print(f"Min score: {min_score}")
                print(f"Max score: {max_score}")
                print(f"Total examples assessed: {len(scores)}")

    print(f"\n{'='*60}")
    print("Completed processing all features!")
    print(f"{'='*60}")
    
    # Print overall summary
    print("\nOverall Summary:")
    for feature, results in all_results.items():
        if results["assessments"]:
            scores = [a["assessment"]["score"] for a in results["assessments"] if isinstance(a["assessment"].get("score"), (int, float))]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"Feature {feature} ({results['direction']}): {avg_score:.2f} avg score ({len(scores)} examples)")
            else:
                print(f"Feature {feature} ({results['direction']}): No valid scores")
        else:
            print(f"Feature {feature} ({results['direction']}): No assessments completed")

# %%
# Run the script
if __name__ == "__main__":
    main()
# %%
