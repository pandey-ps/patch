# credits for app.py: muse spark 1.3
import threading
import time
import gradio as gr
import plotly.graph_objects as go

MODEL_NAME = "PubMedBERT"
MODEL_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

COLUMNS = ["clean", "corrupt", "template", "target", "context_from", "context_to"]

DEFAULT_FACTS = [
    ["Insulin", "Aspirin", "The hormone that regulates blood sugar is {}", "Insulin", "sugar", "pain"],
    ["Aspirin", "Insulin", "The pill often used to reduce fever is {}", "Aspirin", "fever", "sugar"],
    ["Penicillin", "Aspirin", "The first antibiotic discovered to fight infection was {}", "Penicillin", "infection", "pain"],
    ["Vaccine", "Aspirin", "The injection that builds immunity against disease is a {}", "Vaccine", "immunity", "pain"],
    ["Dopamine", "Insulin", "The brain chemical linked to reward is {}", "Dopamine", "brain", "blood"],
    ["Asthma", "Diabetes", "The chronic condition that causes wheezing is {}", "Asthma", "wheezing", "thirst"],
    ["Malaria", "Asthma", "The mosquito-borne disease that causes chills is {}", "Malaria", "mosquito", "pollen"],
    ["Calcium", "Iron", "The mineral that strengthens bones is {}", "Calcium", "bones", "blood"],
    ["Oxygen", "Calcium", "The gas carried by red blood cells is {}", "Oxygen", "blood", "bone"],
    ["Anemia", "Asthma", "The blood disorder caused by low iron is {}", "Anemia", "iron", "dust"],
]


def _rows_to_tuples(rows) -> list[tuple]:
    if rows is None:
        raise gr.Error("Add at least one fact row.")
    if hasattr(rows, "values"):  # gradio passes a pandas DataFrame
        rows = rows.values.tolist()
    facts = []
    for r in rows or []:
        if not r or len(r) < 6 or not r[0] or not r[2]:
            continue
        if "{}" not in str(r[2]):
            raise gr.Error(f"Template must contain {{}}: {r[2]}")
        facts.append(tuple(str(c or "") for c in r[:6]))
    if not facts:
        raise gr.Error("Add at least one fact row.")
    return facts


def _bar(frac: float) -> str:
    pct = max(0, min(100, int(frac * 100)))
    return f"<div class='pbar'><div class='pfill' style='width:{pct}%'></div></div>"


class _TqdmHook:

    def __init__(self):
        self.i = 0
        self.total = 1

    def __call__(self, it, **kw):
        self.total = max(len(it), 1)
        self.i = 0
        for x in it:
            self.i += 1
            yield x


def run_trace(rows):
    from dataset import build_dataset
    import tracer as T

    facts = _rows_to_tuples(rows)
    ds = build_dataset(facts)
    hook = _TqdmHook()
    T.tqdm, _orig = hook, T.tqdm
    box: dict = {}

    def _work():
        try:
            tracer = T.CausalTracer(MODEL_NAME, MODEL_ID)
            result = tracer.trace(ds, show_progress=True)
            tracer.release()
            d = result.to_dict()
            d["num_prompts"] = len(ds)
            box["merged"] = {result.model_name: d}
        except Exception as exc:
            box["error"] = exc
        finally:
            T.tqdm = _orig

    yield _bar(0.03), None, ""
    t = threading.Thread(target=_work, daemon=True)
    t.start()
    while t.is_alive():
        yield _bar(0.05 + 0.9 * hook.i / max(hook.total, 1)), None, ""
        time.sleep(0.3)
    t.join()
    if "error" in box:
        raise box["error"]
    merged = box["merged"]
    fig, table = show_results(merged)
    yield _bar(1.0), fig, table


def show_results(raw):
    (name, d) = next(iter(raw.items()))
    scores = d["mask_patch"]
    peak = int(max(range(len(scores)), key=lambda j: scores[j]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, mode="lines+markers", name=name,
                             line=dict(color="#0d9488", width=2.5),
                             marker=dict(size=6)))
    fig.add_annotation(x=peak, y=scores[peak], text=f"peak L{peak}", showarrow=True)
    fig.update_layout(title="patch restoration by layer",
                      xaxis_title="layer",
                      yaxis_title="avg. prob restoration",
                      template="plotly_dark",
                      font=dict(color="#e2e8f0"),
                      paper_bgcolor="#111c30", plot_bgcolor="#111c30")
    table = ("| model | prompts | peak layer | peak magnitude | clean−corrupt |\n"
             "|---|---|---|---|---|\n"
             f"| {name} | {d['num_prompts']} | L{peak} | {scores[peak]:.2e} | "
             f"{d['clean_minus_corrupt']:+.2e} |")
    return fig, table


with gr.Blocks(title="patch") as demo:
    with gr.Column():
        fact_table = gr.Dataframe(headers=COLUMNS, value=DEFAULT_FACTS,
                                  interactive=True, wrap=True, label="Facts")

    with gr.Column():
        run_btn = gr.Button("Run trace", variant="primary")
        prog_out = gr.HTML("")

    with gr.Column():
        plot_out = gr.Plot(label="Restoration curves")
        table_out = gr.Markdown()

    run_btn.click(run_trace, inputs=fact_table,
                  outputs=[prog_out, plot_out, table_out])

if __name__ == "__main__":
    demo.queue().launch()
