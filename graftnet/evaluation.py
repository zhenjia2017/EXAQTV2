
def answer_presence_gst(entities, gold_answers):
    """
    Compute the answer presence for a set of evidences
    and a parsed answer dict, and return a list of
    answering evidences.
    Return format: (boolean, [evidence-dict, ...])
    """
    """Check whether the given evidence has any of the answers."""
    answer_candidates = entities
    for answer_candidate in answer_candidates:
        # check for year in case the item is a timestamp
        # check if answering candidate
        answer_candidate_id = answer_candidate["id"]
        if candidate_in_answers(answer_candidate_id, gold_answers):
            return True
    return False

def answer_presence(evidences, answers):
    """
    Compute the answer presence for a set of evidences
    and a parsed answer dict, and return a list of
    answering evidences.
    Return format: (boolean, [evidence-dict, ...])
    """
    # initialize
    answer_present = False
    answering_spos = list()

    # go through evidences
    for evidence in evidences:
        if evidence_has_answer(evidence, answers):
            # remember evidence
            answer_present = True
            answering_spos.append(evidence)
    # return results
    return (answer_present, answering_spos)


def evidence_has_answer(evidence, gold_answers):
    """Check whether the given evidence has any of the answers."""
    if not evidence:
        return False
    answer_candidates = evidence["wikidata_entities"]
    for answer_candidate in answer_candidates:
        # check for year in case the item is a timestamp        
        # check if answering candidate
        answer_candidate_id = answer_candidate["id"]
        if candidate_in_answers(answer_candidate_id, gold_answers):
            return True
    return False


def candidate_in_answers(answer_candidate_id, gold_answers):
    """Check if candidate is answer."""
    # get ids
    gold_answer_ids = [answer["id"] for answer in gold_answers]

    # normalize
    answer_candidate_id = answer_candidate_id.lower().strip().replace('"', "").replace("+", "")
    gold_answer_ids = [answer.lower().strip().replace('"', "").replace("+", "") for answer in gold_answer_ids]

    # perform check
    if answer_candidate_id in gold_answer_ids:
        return True

    # no match found
    return False

def mrr_score(answers, gold_answers):
    """Compute MRR score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if candidate_in_answers(answer["answer"], gold_answers):
            return 1.0 / float(answer["rank"])
    return 0.0

def precision_at_1(answers, gold_answers):
    """Compute P@1 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if float(answer["rank"]) > float(1.0):
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0

def hit_at_5(answers, gold_answers):
    """Compute Hit@5 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if float(answer["rank"]) > float(5.0):
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0

def question_is_existential(question):
    existential_keywords = [
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "being",
        "been",
        "did",
        "do",
        "does",
        "done",
        "doing",
        "has",
        "have",
        "had",
        "having",
    ]
    lowercase_question = question.lower()
    lowercase_question = lowercase_question.strip()
    for keyword in existential_keywords:
        if lowercase_question.split()[0] == keyword:
            return True
    return False
