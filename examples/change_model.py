"""演示如何切换 CrossEncoder 模型（需安装 semantic extra）。"""

from subjective_scoring import SubjectiveScoringService

def main() -> None:
    # 同名只加载一份；不同名则 text/code 各一份
    service = SubjectiveScoringService(
        allow_model_load=True,
        text_model="BAAI/bge-reranker-base",
        code_model="BAAI/bge-reranker-base",
    )
    print("text_model=", service.text_model)
    print("code_model=", service.code_model)

if __name__ == "__main__":
    main()
