/* Before/after sliders and the model filter. Plain DOM, no dependencies. */
document.addEventListener("input", (event) => {
  const target = event.target;
  if (target.matches(".ih-compare input[type=range]")) {
    target.closest(".ih-compare").style.setProperty("--pos", target.value + "%");
  } else if (target.matches(".ih-filter")) {
    const query = target.value.trim().toLowerCase();
    document.querySelectorAll(".ih-card").forEach((card) => {
      card.hidden = query !== "" && !card.dataset.search.toLowerCase().includes(query);
    });
  }
});
