// static/js/appointment.js
async function loadAppointmentsAndDoctors() {
	const apptEl = document.getElementById("upcomingList");
	const docContainer = document.getElementById("doctorsContainer");

	// fetch appointments
	const apptRes = await fetch("/api/appointments");
	const apptJson = await apptRes.json();
	apptEl.innerHTML = "";
	if (apptJson.appointments && apptJson.appointments.length) {
		apptJson.appointments.forEach((a) => {
			const li = document.createElement("div");
			li.className = "appointment-card";
			li.innerHTML = `
        <div class="appointment-header">
            <div class="doctor-name">${a.doctor_name}</div>
            <span class="status-tag">${a.status}</span>
        </div>
        <p class="appointment-type">${a.type}</p>
        <div style="margin-top:15px;">
            <div class="appointment-detail"><span class="appointment-detail-icon">🗓️</span><span>${
							a.date
						}</span></div>
            <div class="appointment-detail"><span class="appointment-detail-icon">⏱️</span><span>${
							a.time
						}</span></div>
        </div>
        <div style="font-size: 0.9rem; color: #f8f8f2; margin-top: 15px;">
            <span style="color: #ff79c6;">Reason:</span> ${a.reason || "-"}
        </div>`;
			apptEl.appendChild(li);
		});
	} else {
		apptEl.innerHTML = "<p style='color:#6272a4'>No upcoming appointments.</p>";
	}

	// fetch doctors
	const docRes = await fetch("/api/doctors");
	const docJson = await docRes.json();
	docContainer.innerHTML = "";
	if (docJson.doctors) {
		docJson.doctors.forEach((d) => {
			const card = document.createElement("div");
			card.className = "appointment-card";
			card.style.width = "280px";
			card.style.padding = "15px";
			card.innerHTML = `
        <div style="display:flex;align-items:center;margin-bottom:10px;">
          <div class="profile-avatar" style="background-color:${
						d.color
					};margin-right:15px;">${d.name.charAt(0)}</div>
          <div>
            <div class="doctor-name" style="color:${d.color};">${d.name} (${
				d.specialty
			})</div>
            <div class="profile-role">Next Available: ${d.next_available}</div>
          </div>
        </div>
        <button class="book-btn" style="width:100%;padding:10px;" onclick='bookSlot("${
					d.id
				}")'>Book Now</button>
      `;
			docContainer.appendChild(card);
		});
	}
}

async function bookSlot(doctor_id) {
	const date = prompt("Enter date (YYYY-MM-DD)");
	if (!date) return alert("Booking cancelled.");
	const time = prompt("Enter time (HH:MM)");
	if (!time) return alert("Booking cancelled.");
	const reason = prompt("Reason (optional)") || "";

	const res = await fetch("/api/book", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			doctor_id,
			date,
			time,
			type: "Consultation",
			reason,
		}),
	});
	const json = await res.json();
	if (json.ok) {
		alert("Appointment booked!");
		loadAppointmentsAndDoctors();
	} else {
		alert("Failed to book: " + (json.error || "unknown"));
	}
}

document.addEventListener("DOMContentLoaded", () => {
	// if elements exist on page load
	if (document.getElementById("upcomingList")) {
		loadAppointmentsAndDoctors();
	}
});
