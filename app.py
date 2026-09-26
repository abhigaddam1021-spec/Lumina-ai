import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from data.simulator import get_dataset, DATASETS, DATASET_DISPLAY_NAMES
from nn.network import NeuralNetwork, MODE_CONFIGS, MODE_INFO
from nn.trainer import Trainer

# -- UI Configuration --
st.set_page_config(page_title="DeepMind AI Studio", page_icon="🧠", layout="wide")
st.title("🧠 DeepMind AI Simulation Studio")

# -- Top-level tab navigation --
tab_train, tab_research, tab_compare = st.tabs([
    "🤖 Train on Simulated Data",
    "🌐 Research Mode (Live Internet)",
    "⚖️ Source Comparison",
])


# ===========================================================================
#  TAB 1: Classic training on simulated datasets
# ===========================================================================
with tab_train:
    st.markdown("Train your from-scratch NumPy neural network on simulated datasets!")

    col_sidebar, col_main = st.columns([1, 3])

    with col_sidebar:
        st.header("⚙️ Configuration")
        
        # User-friendly Dataset mapping
        ds_keys = list(DATASETS.keys())
        ds_labels = [DATASET_DISPLAY_NAMES[k] for k in ds_keys]
        selected_ds_label = st.selectbox("1. Choose Dataset (Task)", ds_labels, index=0)
        dataset_name = ds_keys[ds_labels.index(selected_ds_label)]
        
        # User-friendly Mode mapping
        mode_keys = list(MODE_CONFIGS.keys())
        mode_labels = [MODE_INFO[k] for k in mode_keys]
        selected_mode_label = st.selectbox("2. Choose AI Skill Mode", mode_labels, index=mode_keys.index("quick"))
        mode_name = mode_keys[mode_labels.index(selected_mode_label)]
        
        st.markdown("---")
        # Set default values to the IQ 124 winning configuration
        hidden_size  = st.slider("Neurons per Layer", 4, 128, 48, key="hs1")
        epochs       = st.slider("Training Epochs", 10, 500, 150, key="ep1")
        lr           = st.selectbox("Learning Rate", [0.01, 0.005, 0.001, 0.0001], index=1, key="lr1")
        train_button = st.button("🚀 Train AI Now", use_container_width=True, key="tb1")

    with col_main:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Dataset: `{DATASET_DISPLAY_NAMES[dataset_name]}`")
            st.write(f"**Task type:** {DATASETS[dataset_name]['task'].title()}")
            st.write(f"**Active Mode:** {MODE_INFO[mode_name]}")
            config = MODE_CONFIGS[mode_name]
            st.json({
                "Active Layers":    config["depth"],
                "Creativity (Temp)": config["temp"],
                "Daydreaming (Noise)": config["noise"],
                "Dropout Scale":    config["dropout_scale"]
            })

        if train_button:
            with st.spinner("Generating dataset and building AI..."):
                X, y, task, in_size, out_size = get_dataset(dataset_name, n_samples=800)
                net = NeuralNetwork(
                    input_size=in_size, hidden_size=hidden_size,
                    output_size=out_size, task=task, mode=mode_name
                )
                trainer = Trainer(net, epochs=epochs, batch_size=32,
                                  learning_rate=lr, verbose=False)

            st.success(f"Built network with {net.param_count():,} parameters!")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            n_val   = int(len(X) * 0.15)
            X_train, y_train = X[:-n_val], y[:-n_val]
            X_val,   y_val   = X[-n_val:], y[-n_val:]

            start_time = time.time()
            for epoch in range(1, epochs + 1):
                net.set_training(True)
                predictions      = net.forward(X_train)
                loss, grad       = net.compute_loss(y_train, predictions)
                net.backward(grad)
                all_layers = net.hidden_layers[:net._active_layers] + [net.output_layer]
                
                # Apply L2 weight decay (matching benchmark)
                l2_lambda = 1e-4
                for layer in all_layers:
                    layer.dW += l2_lambda * layer.W
                    
                trainer.optimizer.update(all_layers)

                net.set_training(False)
                val_preds        = net.forward(X_val)
                val_loss, _      = net.compute_loss(y_val, val_preds)
                val_acc          = net.accuracy(X_val, y_val)

                progress_bar.progress(epoch / epochs)
                status_text.text(
                    f"Epoch {epoch}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%"
                )

            st.success(f"Training completed in {time.time() - start_time:.2f} seconds!")

            with col2:
                st.subheader("📊 Decision Boundary Visualization")
                if in_size == 2 and out_size == 1:
                    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                         np.arange(y_min, y_max, 0.05))
                    Z = net.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                    fig, ax = plt.subplots()
                    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
                    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
                    st.pyplot(fig)
                else:
                    st.warning(
                        "Decision boundary plotting is only supported for 2D inputs."
                    )


# ===========================================================================
#  TAB 2: Research Mode — live internet data
# ===========================================================================
with tab_research:
    st.markdown(
        "Give the AI **real internet data** to learn from! "
        "Pick two topics, hit Fetch & Train, and the AI will search the web, "
        "read the results, and learn to tell them apart — all in real time."
    )

    try:
        from research.live_data   import fetch_live_dataset, RESEARCH_PRESETS
        from research.searcher    import search, wikipedia_summary
        _RESEARCH_OK = True
    except ImportError as _e:
        st.error(f"Research module not loaded: {_e}")
        _RESEARCH_OK = False

    if _RESEARCH_OK:
        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            st.subheader("⚙️ Research Config")

            preset_name = st.selectbox(
                "Choose a preset theme (or Custom):",
                list(RESEARCH_PRESETS.keys())
            )

            if preset_name == "Custom":
                topic_a = st.text_input("Topic A", value="machine learning")
                topic_b = st.text_input("Topic B", value="climate change")
            else:
                preset_topics = RESEARCH_PRESETS[preset_name]
                topic_a = st.text_input("Topic A", value=preset_topics[0])
                topic_b = st.text_input("Topic B", value=preset_topics[1])

            st.markdown("---")
            results_per  = st.slider("Web results per topic", 3, 15, 6)
            use_wiki     = st.checkbox("Include Wikipedia summaries", value=True)
            vocab_sz     = st.select_slider("Vocabulary size (input features)",
                                            options=[32, 64, 128, 256], value=64)
            hidden_r     = st.slider("Neurons per Layer", 4, 64, 32, key="hs2")
            epochs_r     = st.slider("Training Epochs", 20, 300, 80, key="ep2")
            lr_r         = st.selectbox("Learning Rate",
                                        [0.01, 0.005, 0.001, 0.0001], index=1, key="lr2")
            mode_r       = st.selectbox("AI Mode", list(MODE_CONFIGS.keys()), index=1, key="mode2")
            fetch_button = st.button("🌐 Fetch & Train", use_container_width=True)

            # -- Quick Search Panel --
            st.markdown("---")
            st.subheader("🔍 Quick Search")
            search_query = st.text_input("Search the web:", placeholder="e.g. how do black holes form")
            if st.button("Search", key="srch"):
                if search_query:
                    with st.spinner("Searching..."):
                        results = search(search_query, max_results=5)
                    if results:
                        for r in results:
                            st.markdown(f"**[{r['title']}]({r['url']})**")
                            st.caption(r.get("body", "")[:200])
                            st.divider()
                    else:
                        st.warning("No results found.")

        with col_r2:
            if fetch_button:
                topics = [topic_a.strip(), topic_b.strip()]
                if not all(topics):
                    st.error("Please enter both topics!")
                else:
                    with st.spinner(f"Fetching live data for: {topics} ..."):
                        try:
                            X_r, y_r, class_names, vectorizer, source_log = fetch_live_dataset(
                                topics=topics,
                                results_per_topic=results_per,
                                use_wikipedia=use_wiki,
                                vocab_size=vocab_sz,
                            )
                        except Exception as e:
                            st.error(f"Fetch failed: {e}")
                            st.stop()

                    st.success(
                        f"Fetched {len(X_r)} text samples across {len(class_names)} topics! "
                        f"Input vectors: {X_r.shape[1]} features."
                    )

                    # Show sources
                    with st.expander("📄 Sources fetched from the web"):
                        for topic, logs in source_log.items():
                            st.markdown(f"**{topic.upper()}**")
                            for entry in logs:
                                st.caption(entry)

                    # Show top vocabulary words
                    with st.expander("🔤 Top vocabulary words learned"):
                        if vectorizer.vocabulary_:
                            st.write(", ".join(vectorizer.vocabulary_[:40]))

                    # -- Build & train network on live data --
                    st.subheader("🧠 Training on Live Internet Data")

                    in_sz  = X_r.shape[1]
                    out_sz = len(class_names)
                    task_r = "binary" if out_sz == 2 else "multiclass"

                    # Convert one-hot binary labels [N, 2] to single column [N, 1]
                    if task_r == "binary" and y_r.shape[1] == 2:
                        y_r = y_r[:, 1:2]

                    net_r = NeuralNetwork(
                        input_size=in_sz, hidden_size=hidden_r,
                        output_size=out_sz if out_sz > 2 else 1,
                        task=task_r, mode=mode_r
                    )
                    trainer_r = Trainer(net_r, epochs=epochs_r, batch_size=16,
                                        learning_rate=lr_r, verbose=False)

                    st.info(
                        f"Network: {in_sz} inputs → {net_r.param_count():,} params "
                        f"→ classifying: {class_names}"
                    )

                    n_val_r  = max(2, int(len(X_r) * 0.2))
                    X_tr_r   = X_r[:-n_val_r];  y_tr_r = y_r[:-n_val_r]
                    X_va_r   = X_r[-n_val_r:];  y_va_r = y_r[-n_val_r:]

                    prog_r  = st.progress(0)
                    stat_r  = st.empty()
                    history = {"loss": [], "val_acc": []}

                    for ep in range(1, epochs_r + 1):
                        net_r.set_training(True)
                        preds_r      = net_r.forward(X_tr_r)
                        loss_r, gr   = net_r.compute_loss(y_tr_r, preds_r)
                        net_r.backward(gr)
                        layers_r = net_r.hidden_layers[:net_r._active_layers] + [net_r.output_layer]
                        trainer_r.optimizer.update(layers_r)

                        net_r.set_training(False)
                        vp_r         = net_r.forward(X_va_r)
                        vl_r, _      = net_r.compute_loss(y_va_r, vp_r)
                        va_r         = net_r.accuracy(X_va_r, y_va_r)

                        history["loss"].append(float(loss_r))
                        history["val_acc"].append(float(va_r))

                        prog_r.progress(ep / epochs_r)
                        stat_r.text(
                            f"Epoch {ep}/{epochs_r} | Loss: {loss_r:.4f} | Val Acc: {va_r*100:.1f}%"
                        )

                    final_acc = history["val_acc"][-1] * 100
                    st.success(
                        f"Done! Final validation accuracy: **{final_acc:.1f}%** "
                        f"at distinguishing '{topic_a}' vs '{topic_b}'"
                    )

                    # Loss curve chart
                    st.subheader("📈 Training Loss Curve")
                    fig2, ax2 = plt.subplots(figsize=(7, 3))
                    ax2.plot(history["loss"],   label="Train Loss", color="tomato")
                    ax2.plot(history["val_acc"], label="Val Accuracy", color="steelblue", linestyle="--")
                    ax2.set_xlabel("Epoch")
                    ax2.legend()
                    ax2.set_title("Live Internet Training Progress")
                    st.pyplot(fig2)

                    # Wikipedia preview for each topic
                    st.subheader("📚 Wikipedia Summaries")
                    for t in topics:
                        summary = wikipedia_summary(t)
                        if summary:
                            with st.expander(f"Wikipedia: {t}"):
                                st.write(summary)


# ===========================================================================
#  TAB 3: Source Comparison — Wikipedia vs Britannica vs .gov vs Academic
# ===========================================================================
with tab_compare:
    st.markdown(
        "Research any topic across **four independent authoritative sources** "
        "and let the AI measure how much they agree with each other."
    )

    try:
        from research.source_compare import (
            compare_sources, build_agreement_matrix
        )
        _COMPARE_OK = True
    except ImportError as _e:
        st.error(f"Source compare module error: {_e}")
        _COMPARE_OK = False

    if _COMPARE_OK:
        col_c1, col_c2 = st.columns([1, 2])

        with col_c1:
            st.subheader("⚙️ Compare Config")
            compare_topic = st.text_input(
                "Topic to research:",
                value="black holes",
                placeholder="e.g. climate change, vaccines, democracy, photosynthesis"
            )
            compare_button = st.button("⚖️ Compare All Sources", use_container_width=True)

            st.markdown("---")
            st.markdown("**Sources being compared:**")
            st.markdown("📖 **Wikipedia** — Community-edited encyclopedia")
            st.markdown("📚 **Britannica** — Expert-written since 1768")
            st.markdown("🏛️ **.gov site** — U.S. government agencies (NASA, CDC, NIH...)")
            st.markdown("🔬 **Academic** — Nature, Scientific American, Stanford")

        with col_c2:
            if compare_button and compare_topic.strip():
                results_placeholder = st.empty()

                with st.spinner(f"Fetching '{compare_topic}' from all 4 sources..."):
                    source_results = compare_sources(compare_topic.strip())

                st.success(f"Fetched {sum(1 for r in source_results if r['status'] == 'ok')} / 4 sources successfully!")

                # -- Display each source --
                st.subheader("📋 What each source says")
                for r in source_results:
                    icon   = r.get("icon", "•")
                    name   = r.get("source", "Unknown")
                    url    = r.get("url", "")
                    text   = r.get("text", "")
                    status = r.get("status", "")
                    cred   = r.get("credibility", "")

                    label = f"{icon} {name}"
                    if url:
                        label += f"  — [open article]({url})"

                    with st.expander(label, expanded=True):
                        st.caption(f"*{cred}*")
                        if text:
                            # Show first 600 chars as preview
                            preview = text[:600] + ("..." if len(text) > 600 else "")
                            st.write(preview)
                        else:
                            st.warning(f"Could not retrieve content. Status: `{status}`")

                # -- Agreement matrix --
                st.subheader("🤝 Source Agreement Matrix")
                st.caption(
                    "Measures word-overlap between sources (0 = nothing in common, "
                    "1.0 = identical vocabulary). Higher overlap = more agreement."
                )

                matrix = build_agreement_matrix(source_results)
                source_names = [r["source"].split("(")[0].strip() for r in source_results]

                fig_m, ax_m = plt.subplots(figsize=(5, 4))
                im = ax_m.imshow(matrix, vmin=0, vmax=1, cmap="YlGn")
                ax_m.set_xticks(range(len(source_names)))
                ax_m.set_yticks(range(len(source_names)))
                ax_m.set_xticklabels(source_names, rotation=30, ha="right", fontsize=8)
                ax_m.set_yticklabels(source_names, fontsize=8)
                plt.colorbar(im, ax=ax_m, label="Overlap score")
                # Annotate cells
                for i in range(len(source_names)):
                    for j in range(len(source_names)):
                        ax_m.text(j, i, f"{matrix[i][j]:.2f}",
                                  ha="center", va="center", fontsize=8,
                                  color="black" if matrix[i][j] < 0.7 else "white")
                ax_m.set_title(f"Agreement: '{compare_topic}'")
                plt.tight_layout()
                st.pyplot(fig_m)

                # -- AI's Verdict --
                st.subheader("🧠 AI Verdict")
                ok_sources    = [r for r in source_results if r["status"] == "ok" and r.get("text")]
                failed_sources = [r for r in source_results if r["status"] != "ok" or not r.get("text")]

                if ok_sources:
                    # Find the pair with the highest overlap
                    max_overlap = 0.0
                    best_pair   = (None, None)
                    for i in range(len(source_results)):
                        for j in range(i + 1, len(source_results)):
                            score = matrix[i][j]
                            if score > max_overlap:
                                max_overlap  = score
                                best_pair    = (source_results[i]["source"], source_results[j]["source"])

                    avg_overlap = sum(
                        matrix[i][j]
                        for i in range(len(source_results))
                        for j in range(len(source_results))
                        if i != j
                    ) / max(1, len(source_results) * (len(source_results) - 1))

                    if avg_overlap > 0.25:
                        verdict = (
                            f"✅ **High consensus** across sources "
                            f"(avg overlap: {avg_overlap:.2f}). "
                            f"The topic '{compare_topic}' appears well-documented and consistent."
                        )
                    elif avg_overlap > 0.10:
                        verdict = (
                            f"⚠️ **Moderate agreement** (avg overlap: {avg_overlap:.2f}). "
                            f"Sources broadly agree but use different angles or vocabulary."
                        )
                    else:
                        verdict = (
                            f"❗ **Low agreement** (avg overlap: {avg_overlap:.2f}). "
                            f"Sources may be covering different aspects, or the topic is contested."
                        )

                    st.markdown(verdict)
                    if best_pair[0]:
                        st.info(
                            f"Most aligned pair: **{best_pair[0]}** and **{best_pair[1]}** "
                            f"(overlap: {max_overlap:.2f})"
                        )

                if failed_sources:
                    st.warning(
                        "Could not retrieve: " +
                        ", ".join(r["source"] for r in failed_sources) +
                        ". Some sites block automated access."
                    )

