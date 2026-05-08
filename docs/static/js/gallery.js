(async function () {
  const objSelect = document.getElementById('obj-select');
  const gripperSelect = document.getElementById('gripper-select');
  const video = document.getElementById('cmp-video');
  const caption = document.getElementById('cmp-caption');
  if (!objSelect || !gripperSelect || !video) return;

  let entries;
  try {
    const res = await fetch('static/data/manifest.json');
    entries = await res.json();
  } catch (err) {
    caption.textContent = 'Failed to load comparison gallery manifest.';
    console.error(err);
    return;
  }

  const byObject = new Map();
  for (const e of entries) {
    if (!byObject.has(e.object)) byObject.set(e.object, []);
    byObject.get(e.object).push(e);
  }

  for (const [obj, items] of byObject) {
    const opt = document.createElement('option');
    opt.value = obj;
    opt.textContent = items[0].label;
    objSelect.appendChild(opt);
  }

  function loadVideo() {
    const obj = objSelect.value;
    const gripper = gripperSelect.value;
    const entry = byObject.get(obj).find(e => e.gripper === gripper);
    if (!entry) return;
    video.src = 'static/videos/comparisons/' + entry.file;
    caption.textContent = `${entry.label} · ${entry.gripper_label} — left: GT mesh, right: BakedSDF reconstruction`;
  }

  function repopulateGrippers() {
    gripperSelect.innerHTML = '';
    for (const e of byObject.get(objSelect.value)) {
      const opt = document.createElement('option');
      opt.value = e.gripper;
      opt.textContent = e.gripper_label;
      gripperSelect.appendChild(opt);
    }
    loadVideo();
  }

  objSelect.addEventListener('change', repopulateGrippers);
  gripperSelect.addEventListener('change', loadVideo);
  repopulateGrippers();
})();
