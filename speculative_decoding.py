import random
import math

def speculative_decode(prompt, fast_model, large_model, max_new_tokens, block_size=5):
    """
    Perform speculative decoding by combining a fast proposal model with a larger verification model.
    
    Parameters:
        prompt (str or list): The initial text (or list of tokens) to seed decoding.
        fast_model (object): An object that supports:
            - generate_candidates(context, block_size): returns a tuple 
                (candidate_tokens, candidate_log_probs)
                where candidate_tokens is a list of tokens proposed from the context,
                and candidate_log_probs is a list of the corresponding log probabilities.
        large_model (object): An object that supports:
            - verify_candidates(context, candidate_tokens): returns a list of log probabilities 
                for each token in candidate_tokens, based on the provided context.
            - sample_next(context): returns a single token sampled from the model 
                given the current context.
        max_new_tokens (int): Maximum number of tokens to append to the prompt.
        block_size (int): Number of tokens that the fast model will propose in one block.
        
    Returns:
        list: The list of tokens corresponding to the prompt with new tokens appended.
    
    Algorithm:
        1. Begin with the given prompt.
        2. While fewer than max_new_tokens have been generated:
             a. Ask the fast model for a block of candidate tokens (and their log probabilities).
             b. With one call, have the large model verify the entire block by returning its own
                log probabilities for those tokens.
             c. For each candidate in the block, compute the acceptance ratio:
                  ratio = exp(large_log_prob - fast_log_prob)
                and set the acceptance probability to min(1, ratio).
             d. For each candidate token, draw a random number. If the number is below the 
                acceptance probability, accept that token and move to the next; otherwise, break.
             e. If at least one token in the block is accepted, append them to the decoded sequence.
                If the block is truncated (i.e. not all candidates are accepted), generate one token 
                from the large model as a fallback.
             f. If no token in the candidate block is accepted at all, fall back to generating one token 
                using the large model.
        3. Return the final decoded sequence.
    """
    # If prompt is a string, split into tokens; otherwise assume it is already a list of tokens.
    decoded = prompt.split() if isinstance(prompt, str) else prompt[:]
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        # 1. Fast model produces a candidate block with its log probabilities.
        candidate_tokens, candidate_log_probs = fast_model.generate_candidates(decoded, block_size)
        if not candidate_tokens:
            break  # No candidates were generated

        # 2. Verify the entire candidate block in one large model call.
        large_log_probs = large_model.verify_candidates(decoded, candidate_tokens)

        accepted_tokens = []
        # 3. For each candidate token, check whether to accept it.
        for token, fast_lp, large_lp in zip(candidate_tokens, candidate_log_probs, large_log_probs):
            # Calculate acceptance ratio (in probability space)
            ratio = math.exp(large_lp - fast_lp)
            acceptance_prob = min(1.0, ratio)
            # Draw a uniform random value and decide acceptance.
            if random.random() < acceptance_prob:
                accepted_tokens.append(token)
            else:
                # Stop processing further tokens in this block on first rejection.
                break

        # 4. Append accepted tokens or use fallback generation with the large model.
        if accepted_tokens:
            decoded.extend(accepted_tokens)
            tokens_generated += len(accepted_tokens)
            # If not all candidates were accepted, generate one token via large_model as fallback.
            if len(accepted_tokens) < len(candidate_tokens):
                fallback_token = large_model.sample_next(decoded)
                decoded.append(fallback_token)
                tokens_generated += 1
        else:
            # If no tokens were accepted, use large_model to generate one token directly.
            fallback_token = large_model.sample_next(decoded)
            decoded.append(fallback_token)
            tokens_generated += 1

    return decoded
