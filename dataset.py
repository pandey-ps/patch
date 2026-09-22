from typing import Optional

def build_dataset(
    facts: list[tuple],
    variants: Optional[list[tuple[str, str]]] = None,
) -> list[dict]:
    if variants is None:
        variants = [
            ("", ""),
            ("In medicine, ", "."),
            ("Clinically, ", "."),
            ("Quick check: ", "."),
            ("Med 101: ", "."),
        ] #just to multiply paraphrase

    dataset: list[dict] = []

    for fact in facts:
        clean_sub, corr_sub, template, target, ctx_from, ctx_to = fact
        for prefix, suffix in variants:
            var_template = f"{prefix}{template}{suffix}"
            mask_prompt = var_template.format("[MASK]")
            corrupt_template = var_template.replace(ctx_from, ctx_to, 1)
            corrupt_mask_prompt = corrupt_template.format("[MASK]")

            dataset.append(
                {
                    "clean_subject": clean_sub,
                    "corrupt_subject": corr_sub,
                    "template": var_template,
                    "context_from": ctx_from,
                    "context_to": ctx_to,
                    "clean_prompt": var_template.format(clean_sub),
                    "corrupt_prompt": var_template.format(corr_sub),
                    "mask_prompt": mask_prompt,
                    "corrupt_mask_prompt": corrupt_mask_prompt,
                    "target": target,
                }
            )

    return dataset
