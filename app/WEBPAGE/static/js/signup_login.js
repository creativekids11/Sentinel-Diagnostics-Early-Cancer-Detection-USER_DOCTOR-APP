// static/js/signup_login.js

async function ajaxSignup(e) {
	e.preventDefault();
	const form = e.target;
	const data = {
		fullname: form.fullname.value,
		username: form.username.value,
		password: form.password.value,
		role: form.role.value,
		contact: form.contact.value,
	};
	const res = await fetch("/api/signup", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(data),
	});
	const j = await res.json();
	if (j.ok && j.redirect) window.location = j.redirect;
	else alert("Signup error: " + (j.error || "unknown"));
}

async function ajaxLogin(e) {
	e.preventDefault();
	const f = e.target;
	const username = f.querySelector('input[name="username"]').value;
	const password = f.querySelector('input[name="password"]').value;
	const res = await fetch("/api/login", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ username, password }),
	});
	const j = await res.json();
	if (j.ok && j.redirect) window.location = j.redirect;
	else alert(j.error || "Login failed");
}

async function submitRisk(e) {
	e.preventDefault();
	const form = e.target;
	const assessment = form.dataset.assessment || "breast";
	const answers = {};
	// simple: gather selects/inputs with name attributes
	new FormData(form).forEach((v, k) => (answers[k] = v));
	const res = await fetch("/api/risk", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ assessment, answers }),
	});
	const j = await res.json();
	if (j.ok) {
		alert(`Risk score: ${j.score} (${j.level})`);
	} else {
		alert("Failed to compute risk.");
	}
}

document.addEventListener("DOMContentLoaded", () => {
	const signupForm = document.getElementById("signupForm");
	if (signupForm) signupForm.addEventListener("submit", ajaxSignup);

	const loginForm = document.getElementById("loginForm");
	if (loginForm) loginForm.addEventListener("submit", ajaxLogin);

	const riskForm = document.getElementById("riskForm");
	if (riskForm) riskForm.addEventListener("submit", submitRisk);
});
