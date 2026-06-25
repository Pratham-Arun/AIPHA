def export_graph_visualization(workflow):
    """
    Exports the LangGraph workflow to a Mermaid markdown file.
    In a real environment, you can render this to a PNG.
    """
    try:
        mermaid_code = workflow.graph.get_graph().draw_mermaid()
        with open("workflow.mmd", "w") as f:
            f.write(mermaid_code)
        print("  Workflow Visual  : Exported to workflow.mmd")
    except Exception as e:
        print(f"  Workflow Visual  : Failed to export ({e})")
