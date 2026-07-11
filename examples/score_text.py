"""文本主观题评分示例（默认不加载大模型）。"""

from subjective_scoring import SubjectiveScoringService

def main() -> None:
    service = SubjectiveScoringService(allow_model_load=False)
    result = service.score({
        "question_id": "q1",
        "max_score": 10,
        "scoring_mode": "text",
        "student_answer": "索引可以让数据库查得更快，少做全表扫描。",
        "reference_answer": "索引可以提高查询效率，减少全表扫描。",
        "scoring_points": [
            {"id": "p1", "text": "提高查询效率", "score": 5, "required": True},
            {"id": "p2", "text": "减少全表扫描", "score": 5},
        ],
    })
    print("score=", result.score)
    print("confidence=", result.confidence)
    print("review=", result.review_level)
    print("track=", result.track)
    for p in result.matched_points:
        print(" matched", p.point_id, p.score, p.reason)
    for p in result.missed_points:
        print(" missed", p.point_id, p.reason)

if __name__ == "__main__":
    main()
