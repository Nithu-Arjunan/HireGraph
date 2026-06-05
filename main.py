from unittest import result
from pprint import pprint
from langgraph.types import Command

from src.graph import build_hiregraph

def main():
    graph = build_hiregraph()

    initial_state = {
        "resume_path": "sample_data/resume/resume_eitan.md",
        "jd_path": "sample_data/jd/jd_junior_data.md",
    }

    config = {
        "configurable": {
            "thread_id": "candidate-review-001"
        }
    }
   

    result = graph.invoke(initial_state, config=config)


    # print("\n========== RESUME TEXT ==========")
    # print(result["raw_resume"][:1000])

    # print("\n========== JD TEXT ==========")
    # print(result["raw_jd"][:1000])

    # print(result["jd_requirements"])
    # # print(result["required_skills"])

    # print("\n========== SKILL EVALUATIONS ==========")
    # print(result.get("skill_evaluations", []))

    # print("\n========== DIMENSION EVALUATIONS ==========")
    # print(result.get("dimension_evaluations", []))

    # print("\n========== SCORE SUMMARY ==========")
    # print(result.get("score_summary", {}))

    # print("\n========== FINAL SCORE ==========")
    # print(result.get("final_score"))


    # If graph paused for human review
    if "__interrupt__" in result:
        print("\n========== HUMAN REVIEW REQUIRED ==========")
        pprint(result["__interrupt__"])

        decision = input("\nApprove or reject candidate? Type approved/rejected: ").strip().lower()
        notes = input("Reviewer notes: ").strip()

        reviewer_input = {
            "decision": decision,
            "notes": notes
        }

        result = graph.invoke(
            Command(resume=reviewer_input),
            config=config,
        )

    print("\n========== FINAL RESULT ==========")
    pprint({
        "recommendation": result.get("recommendation"),
        "human_review_decision": result.get("human_review_decision"),
        "final_score": result.get("final_score"),
        "candidate_email": result.get("candidate_email"),
        "sender_email": result.get("sender_email"),
    })

    print("\n========== RECOMMENDATION ==========")
    print(result.get("recommendation"))
    print(result.get("recommendation_reasoning"))

    print("\n========== EMAIL ==========")
    print("To:", result.get("candidate_email"))
    print("From:", result.get("sender_email"))
    print(result.get("draft_email"))

    print("\n========== REJECTION ==========")
    print("To:", result.get("candidate_email"))
    print("From:", result.get("sender_email"))
    print(result.get("rejection_email"))

    print("\n========== ACTION STATUS ==========")
    print("email_sent:", result.get("email_sent"))
    print("ats_updated:", result.get("ats_updated"))
    print("rejection_logged:", result.get("rejection_logged"))
    print("compensation_done:", result.get("compensation_done"))

    print("\n========== AUDIT TRAIL ==========")
    for item in result.get("audit_trail", []):
        print(item)

    ### For visualization
    from src.graph_print import save_graph_png, show_graph
    save_graph_png(graph, "hiregraph")
    show_graph("hiregraph")



if __name__ == "__main__":
    main()
