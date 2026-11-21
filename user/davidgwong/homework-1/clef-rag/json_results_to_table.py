import json

# Example input containing multiple JSON objects
raw_input = """
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 1,
  "retrieval_passed": 11,
  "retrieval_eligible": 12,
  "retrieval_checked": 11,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9090909090909091,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9090909090909091,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.3939393939393939,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19696969696969696,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 11,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 1
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 11,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 11,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 11,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 11,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 11,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 11,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
{
  "cases_run": 20,
  "errors": 0,
  "retrieval_passed": 12,
  "retrieval_eligible": 12,
  "retrieval_checked": 12,
  "retrieval_metrics": {
    "1": {
      "precision": 1.0,
      "recall": 0.9166666666666666,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 0.9166666666666666,
      "ndcg": 1.0
    },
    "3": {
      "precision": 0.38888888888888884,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    },
    "6": {
      "precision": 0.19444444444444442,
      "recall": 1.0,
      "hit_rate": 1.0,
      "mrr": 1.0,
      "map": 1.0,
      "ndcg": 1.0
    }
  },
  "catalog_passed": 4,
  "abstention_passed": 4,
  "answerable_questions_supported": 12,
  "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
  "retrieval_unscored": 0
}
"""
def parse_safely(text):
    decoder = json.JSONDecoder()
    text = text.strip()
    idx = 0
    objects = []
    
    while idx < len(text):
        # Skip whitespaces and blank lines
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
            
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            objects.append(obj)
            idx = end_idx
        except json.JSONDecodeError as e:
            print(f"Error parsing near character position {idx}: {e}")
            break
            
    return objects

data_list = parse_safely(raw_input)

# Standard metrics extraction
metric_keys = ["precision", "recall", "hit_rate", "mrr", "map", "ndcg"]
headers = []
rows = []

for data in data_list:
    metrics = data.get("retrieval_metrics", {})
    k_values = sorted(metrics.keys(), key=lambda x: int(x))
    
    if not headers:
        for k in k_values:
            for metric in metric_keys:
                header_name = f"{metric.capitalize() if metric not in ['mrr', 'map', 'ndcg'] else metric} @ {k}"
                headers.append(header_name)
    
    row_values = []
    for k in k_values:
        for metric in metric_keys:
            val = metrics.get(str(k), {}).get(metric, "")
            row_values.append(str(val))
            
    rows.append(row_values)

print("\t".join(headers))
for row in rows:
    print("\t".join(row))