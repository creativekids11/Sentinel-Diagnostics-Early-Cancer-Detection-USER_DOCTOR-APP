// Lightweight calendar rendering for patient dashboard
(function(){
  const daysEl = document.getElementById('calendarDays');
  const monthLabel = document.getElementById('calendarMonthLabel');
  const prevBtn = document.getElementById('prevMonthBtn');
  const nextBtn = document.getElementById('nextMonthBtn');
  if(!daysEl || !monthLabel) return;

  let today = new Date();
  let view = new Date(today.getFullYear(), today.getMonth(), 1);
  let selected = new Date(today.getFullYear(), today.getMonth(), today.getDate());

  function formatMonth(date){
    const fmt = new Intl.DateTimeFormat(undefined, { month: 'long', year: 'numeric' });
    return fmt.format(date);
  }

  function buildDays(date){
    daysEl.innerHTML = '';
    monthLabel.textContent = formatMonth(date);

    const start = new Date(date.getFullYear(), date.getMonth(), 1);
    const end = new Date(date.getFullYear(), date.getMonth() + 1, 0);

    const startDay = start.getDay();
    const total = end.getDate();

    // Fill previous month placeholders
    const prevEnd = new Date(date.getFullYear(), date.getMonth(), 0).getDate();
    for(let i = startDay - 1; i >= 0; i--){
      const d = document.createElement('div');
      d.className = 'day muted';
      d.textContent = String(prevEnd - i);
      daysEl.appendChild(d);
    }

    // Current month
    for(let dnum = 1; dnum <= total; dnum++){
      const d = document.createElement('div');
      d.className = 'day';
      d.textContent = String(dnum);
      const thisDate = new Date(date.getFullYear(), date.getMonth(), dnum);
      if(sameDate(thisDate, today)) d.classList.add('today');
      if(sameDate(thisDate, selected)) d.classList.add('selected');
      d.addEventListener('click', () => {
        selected = thisDate;
        buildDays(view);
        const ev = new CustomEvent('dashboard:daySelected', { detail: { date: thisDate } });
        window.dispatchEvent(ev);
      });
      daysEl.appendChild(d);
    }

    // Fill next month placeholders to complete 6 rows view
    const filled = daysEl.children.length;
    const need = Math.max(0, 42 - filled);
    for(let i = 1; i <= need; i++){
      const d = document.createElement('div');
      d.className = 'day muted';
      d.textContent = String(i);
      daysEl.appendChild(d);
    }
  }

  function sameDate(a,b){
    return a.getFullYear()===b.getFullYear() && a.getMonth()===b.getMonth() && a.getDate()===b.getDate();
  }

  prevBtn && prevBtn.addEventListener('click', () => {
    view.setMonth(view.getMonth() - 1);
    buildDays(view);
  });
  nextBtn && nextBtn.addEventListener('click', () => {
    view.setMonth(view.getMonth() + 1);
    buildDays(view);
  });

  buildDays(view);
})();
