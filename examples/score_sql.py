"""SQL 结构评分示例。"""

from subjective_scoring import SubjectiveScoringService

def main() -> None:
    service = SubjectiveScoringService(allow_model_load=False)
    result = service.score({
        "question_id": "s1",
        "max_score": 10,
        "scoring_mode": "sql",
        "reference_answer": "SELECT name FROM student WHERE age > 18",
        "student_answer": "select name from student where age > 18",
    })
    print("score=", result.score, "track=", result.track)
    print("warnings=", result.warnings)

if __name__ == "__main__":
    main()
