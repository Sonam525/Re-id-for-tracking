/* Page configuration. */
const CONFIG = {
  datasetRepo: "Sonam5/Calf-Identity-Tracking-Masks",
  videoBase: "https://huggingface.co/datasets/Sonam5/Calf-Identity-Tracking-Masks/resolve/main/videos/",
  // phone versions of the side-by-side clips: left panel above right panel
  stacked: {
    refine: { src: "media/20240422_1641-1748_refine_stacked.mp4", poster: "media/refine_poster_stacked.jpg" },
    pair: { src: "20240423_1800-2400_teacher_student_stacked.mp4", poster: "media/teacher_student_poster_stacked.jpg", remote: true },
  },
  // one video per day; "mosaic" is the minutes mosaicked around the times people were in the pen
  week: [
    { day: "04-22", file: "20240422", role: "validation", from: "12:45", mosaic: 26 },
    { day: "04-23", file: "20240423", role: "validation", from: "05:44", mosaic: 151 },
    { day: "04-24", file: "20240424", role: "held-out test", from: "05:48", mosaic: 91 },
    { day: "04-25", file: "20240425", role: "held-out test", from: "05:44", mosaic: 123 },
    { day: "04-26", file: "20240426", role: "held-out test", from: "05:44", mosaic: 181 },
    { day: "04-27", file: "20240427", role: "held-out test", from: "05:45", mosaic: 39 },
    { day: "04-28", file: "20240428", role: "held-out test", from: "05:40", mosaic: 49 },
  ],
  weekStart: "04-24",
};

/* On a phone the side-by-side panels are too small, so use the stacked clips. */
function useStackedClips() {
  if (!window.matchMedia("(max-width: 640px)").matches) return;
  for (const [id, alt] of Object.entries(CONFIG.stacked)) {
    const v = document.getElementById(id);
    if (!v) continue;
    v.querySelector("source").src = alt.remote ? CONFIG.videoBase + alt.src : alt.src;
    v.poster = alt.poster;
    v.width = 640; v.height = 784;
    v.load();
  }
}

/* The week: one player, one button per day. */
function weekCaption(d) {
  return `${d.day}, ${d.role}, from ${d.from}. Mosaicked for ${d.mosaic} minutes in all, around the times people were in the pen.`;
}
function setupWeek() {
  const box = document.getElementById("daypick"), v = document.getElementById("weekvideo"), cap = document.getElementById("week-cap");
  if (!box || !v) return;
  const buttons = CONFIG.week.map(d => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = d.day; b.setAttribute("aria-pressed", String(d.day === CONFIG.weekStart));
    b.addEventListener("click", () => {
      buttons.forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      v.pause();
      v.querySelector("source").src = CONFIG.videoBase + "week/" + d.file + "_identities.mp4";
      v.poster = "media/week/" + d.file + ".jpg";
      v.setAttribute("aria-label", `Calf identities on 2024-${d.day}, 50 times speed`);
      cap.textContent = weekCaption(d);
      v.load();
    });
    box.appendChild(b);
    return b;
  });
}

document.querySelectorAll("[data-dataset-link]").forEach(a => { a.href = "https://huggingface.co/datasets/" + CONFIG.datasetRepo; });
useStackedClips();
setupWeek();
