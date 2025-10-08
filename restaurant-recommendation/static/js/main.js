// ======================= main.js (updated) =======================

// ⭐ Cuisine-based image mapping
function getCuisineImage(cuisine) {
  if (!cuisine) return null;
  const c = cuisine.toLowerCase();

  if (c.includes("pizza")) return "/static/images/pizza.jpg";
  if (c.includes("indian")) return "/static/images/curry.jpg";
  if (c.includes("chinese") || c.includes("dimsum")) return "/static/images/noodles.jpg";
  if (c.includes("mexican") || c.includes("tacos")) return "/static/images/tacos.jpg";
  if (c.includes("burger") || c.includes("american")) return "/static/images/burger.jpg";
  if (c.includes("italian") || c.includes("pasta") || c.includes("lasagna")) return "/static/images/pasta.jpg";
  if (c.includes("japanese") || c.includes("sushi") || c.includes("ramen")) return "/static/images/sushi.jpg";
  if (c.includes("ice cream") || c.includes("dessert")) return "/static/images/icecream.jpg";
  if (c.includes("cafe") || c.includes("coffee")) return "/static/images/coffee.jpg";
  if (c.includes("bbq") || c.includes("grill") || c.includes("steak")) return "/static/images/bbq.jpg";
  if (c.includes("bakery") || c.includes("cake") || c.includes("pastry")) return "/static/images/cake.jpg";

  if (c.includes("beverages") || c.includes("juices")) return "/static/images/drinks.jpg";
  if (c.includes("international") || c.includes("fusion")) return "/static/images/worldfood.jpg";
  if (c.includes("french") || c.includes("european")) return "/static/images/french.jpg";
  if (c.includes("brazilian") || c.includes("latin")) return "/static/images/brazil.jpg";
  if (c.includes("seafood") || c.includes("prawn")) return "/static/images/seafood.jpg";
  if (c.includes("shawarma") || c.includes("lebanese") || c.includes("arabian")) return "/static/images/arabian.jpg";
  if (c.includes("african") || c.includes("ethiopian")) return "/static/images/african.jpg";
  if (c.includes("vegan") || c.includes("salad") || c.includes("healthy")) return "/static/images/vegan.jpg";
  if (c.includes("breakfast") || c.includes("pancake") || c.includes("brunch")) return "/static/images/breakfast.jpg";

  if (c.includes("street food") || c.includes("snacks")) return "/static/images/food1.jpg";
  if (c.includes("diet") || c.includes("protein")) return "/static/images/food2.jpg";
  if (c.includes("thai") || c.includes("asian") || c.includes("korean")) return "/static/images/food3.jpg";
  if (c.includes("spanish") || c.includes("greek") || c.includes("mediterranean")) return "/static/images/worldfood.jpg";

  return null;
}

// ⭐ City-based background mapping
function getCityImage(city) {
  if (!city) return null;
  const c = city.toLowerCase();

  if (c.includes("hyderabad")) return "/static/images/hyderabad.jpg";
  if (c.includes("mumbai")) return "/static/images/mumbai.jpg";
  if (c.includes("delhi")) return "/static/images/delhi.jpg";
  if (c.includes("kolkata")) return "/static/images/kolkata.jpg";
  if (c.includes("chennai")) return "/static/images/chennai.jpg";
  if (c.includes("bengaluru") || c.includes("bangalore")) return "/static/images/banglore.jpg";
  if (c.includes("sao paulo")) return "/static/images/brazil.jpg";

  return null;
}

// ⭐ Stock fallback images
const stockImages = [
  "/static/images/food1.jpg",
  "/static/images/food2.jpg",
  "/static/images/food3.jpg"
];

// ⭐ Choose best image (smarter priority)
function getRestaurantImage(r) {
  // 1. Use actual restaurant image if available
  if (r.Image_URL && r.Image_URL.trim() !== "") {
    return r.Image_URL;
  }

  // 2. City-specific fallback
  const cityImg = getCityImage(r.City || r.city);
  if (cityImg) return cityImg;

  // 3. Cuisine-specific fallback
  const cuisineImg = getCuisineImage(r.Cuisines || r.cuisines);
  if (cuisineImg) return cuisineImg;

  // 4. Random stock image
  return stockImages[Math.floor(Math.random() * stockImages.length)];
}

// ⭐ Wishlist functions
function toggleWishlist(name) {
  let wishlist = JSON.parse(localStorage.getItem("wishlist")) || [];
  if (wishlist.includes(name)) {
    wishlist = wishlist.filter(r => r !== name);
  } else {
    wishlist.push(name);
  }
  localStorage.setItem("wishlist", JSON.stringify(wishlist));
  renderWishlist();
}

function renderWishlist() {
  const wishlistDiv = document.getElementById("wishlist");
  let wishlist = JSON.parse(localStorage.getItem("wishlist")) || [];
  if (!wishlistDiv) return;
  wishlistDiv.innerHTML =
    wishlist.length === 0
      ? "<p>No restaurants in wishlist yet ❤️</p>"
      : "<h2>My Wishlist ❤️</h2>" + wishlist.map(n => `<p>⭐ ${n}</p>`).join("");
}

// =================== Restaurants (fetch & render) ===================
function loadRestaurants(page = 1) {
  // support query from URL (?search=...)
  const urlParams = new URLSearchParams(window.location.search);
  const search = (document.getElementById("search")?.value || urlParams.get('search') || "").trim();
  const city = document.getElementById("cityFilter")?.value || "";
  const cuisine = document.getElementById("cuisineFilter")?.value || "";
  const rating = document.getElementById("ratingFilter")?.value || "";
  const sort = document.getElementById("sortFilter")?.value || "";

  const params = new URLSearchParams();
  if (search) params.append("search", search);
  if (city) params.append("city", city);
  if (cuisine) params.append("cuisine", cuisine);
  if (rating) params.append("rating", rating);
  if (sort) params.append("sort", sort);
  params.append("page", page);

  const loadingEl = document.getElementById("loading");
  if (loadingEl) loadingEl.style.display = "block";

  fetch(`/api/restaurants?${params.toString()}`)
    .then(res => res.json())
    .then(data => {
      if (loadingEl) loadingEl.style.display = "none";
      const list = document.getElementById("restaurantGrid");
      const pagination = document.getElementById("pagination");
      if (list) list.innerHTML = "";
      if (pagination) pagination.innerHTML = "";

      if (!data || !data.restaurants || data.restaurants.length === 0) {
        if (list) list.innerHTML = "<p>No restaurants found.</p>";
        return;
      }

      data.restaurants.forEach(r => {
        const image = getRestaurantImage(r);
        const html = `
          <div class="card" style="position:relative;">
            <span class="wishlist-icon" data-liked="false" onclick="toggleHeart(this, '${escapeJsString(r["Restaurant Name"] || "")}')">❤️</span>
            <img src="${image}" alt="${escapeHtml(r["Restaurant Name"] || "Restaurant image")}" loading="lazy" onerror="this.onerror=null;this.src='/static/images/restaurant-default.jpg';">
            <div class="card-content">
              <h3>${escapeHtml(r["Restaurant Name"] || "")}</h3>
              <p><b>City:</b> ${escapeHtml(r.City || "")}</p>
              <p><b>Cuisine:</b> ${escapeHtml(r.Cuisines || "")}</p>
              <p><b>Rating:</b> ⭐ ${escapeHtml(r["Aggregate rating"] || "")} ${escapeHtml(r["Rating text"] || "")}</p>
              <p><b>Cost for two:</b> ${escapeHtml(String(r["Average Cost for two"] || ""))} ${escapeHtml(r.Currency || "")}</p>
              <p><b>Votes:</b> ${escapeHtml(String(r.Votes || ""))}</p>
            </div>
          </div>`;
        if (list) list.insertAdjacentHTML("beforeend", html);
      });

      renderPagination(Math.ceil((data.total || data.restaurants.length) / (data.per_page || data.restaurants.length)), data.page || 1);
    })
    .catch(err => {
      console.error("Error loading restaurants:", err);
      const list = document.getElementById("restaurantGrid");
      if (list) list.innerHTML = "<p>Error loading restaurants.</p>";
      if (loadingEl) loadingEl.style.display = "none";
    });
}

// helper to escape HTML (simple)
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
// helper to escape for JS single-quoted string
function escapeJsString(str) {
  return String(str).replace(/'/g, "\\'").replace(/\n/g, "\\n");
}

// =================== Pagination ===================
function renderPagination(totalPages, currentPage) {
  const pagination = document.getElementById("pagination");
  if (!pagination) return;
  pagination.innerHTML = "";

  if (currentPage > 1) {
    const prev = document.createElement("button");
    prev.textContent = "« Prev";
    prev.onclick = () => loadRestaurants(currentPage - 1);
    pagination.appendChild(prev);
  }

  if (currentPage > 3) {
    addPageButton(1, currentPage, pagination);
    if (currentPage > 4) pagination.appendChild(makeDots());
  }

  for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
    addPageButton(i, currentPage, pagination);
  }

  if (currentPage < totalPages - 2) {
    if (currentPage < totalPages - 3) pagination.appendChild(makeDots());
    addPageButton(totalPages, currentPage, pagination);
  }

  if (currentPage < totalPages) {
    const next = document.createElement("button");
    next.textContent = "Next »";
    next.onclick = () => loadRestaurants(currentPage + 1);
    pagination.appendChild(next);
  }
}

function addPageButton(page, currentPage, container) {
  const btn = document.createElement("button");
  btn.textContent = page;
  if (page === currentPage) btn.classList.add("active");
  btn.onclick = () => loadRestaurants(page);
  container.appendChild(btn);
}

function makeDots() {
  const span = document.createElement("span");
  span.textContent = "...";
  span.style.margin = "0 6px";
  return span;
}

// =================== Recommendations (slider) ===================

let recAutoInterval = null;
const REC_SCROLL_AMOUNT = 300;
const REC_SCROLL_DEBOUNCE = 300; // ms, after smooth scroll to update buttons

function loadRecommendations() {
  fetch("/api/recommendations")
    .then(res => res.json())
    .then(data => {
      const slider = document.getElementById("recSlider");
      if (!slider) return;
      slider.innerHTML = "";

      // Build recommendation cards
      (data || []).forEach(r => {
        const image = getRestaurantImage(r);
        const safeName = escapeHtml(r["Restaurant Name"] || "");
        const html = `
          <div class="rec-card" data-name="${escapeJsString(r["Restaurant Name"] || "")}">
            <img src="${image}" alt="${safeName}" loading="lazy" onerror="this.onerror=null;this.src='/static/images/restaurant-default.jpg';">
            <h4>${safeName}</h4>
            <p><b>City:</b> ${escapeHtml(r.City || "")}</p>
            <p><b>Cuisine:</b> ${escapeHtml(r.Cuisines || "")}</p>
            <p><b>Rating:</b> ⭐ ${escapeHtml(r["Aggregate rating"] || "")}</p>
          </div>`;
        slider.insertAdjacentHTML("beforeend", html);
      });

      // Initialize/re-init slider behaviour AFTER DOM insertion
      initRecSlider();
    })
    .catch(err => {
      console.error("Error loading recommendations:", err);
    });
}

function initRecSlider() {
  const slider = document.getElementById("recSlider");
  const leftBtn = document.querySelector(".scroll-btn.left");
  const rightBtn = document.querySelector(".scroll-btn.right");

  if (!slider) return;

  // Ensure buttons exist
  if (!leftBtn || !rightBtn) return;

  // Remove previous listeners to avoid duplication
  slider.removeEventListener("scroll", updateScrollButtons);
  slider.removeEventListener("mouseenter", stopAutoScroll);
  slider.removeEventListener("mouseleave", startAutoScroll);
  leftBtn.removeEventListener("click", scrollLeft);
  rightBtn.removeEventListener("click", scrollRight);

  // Attach listeners
  leftBtn.addEventListener("click", scrollLeft);
  rightBtn.addEventListener("click", scrollRight);
  slider.addEventListener("scroll", updateScrollButtons);
  slider.addEventListener("mouseenter", stopAutoScroll);
  slider.addEventListener("mouseleave", startAutoScroll);

  // initial update & start auto-scroll if many cards
  setTimeout(() => {
    updateScrollButtons();
    startAutoScroll();
  }, 100); // small timeout to ensure layout metrics are ready
}

function updateScrollButtons() {
  const slider = document.getElementById("recSlider");
  const leftBtn = document.querySelector(".scroll-btn.left");
  const rightBtn = document.querySelector(".scroll-btn.right");
  if (!slider || !leftBtn || !rightBtn) return;

  // if content fits, hide both
  if (slider.scrollWidth <= slider.clientWidth + 2) {
    leftBtn.style.display = "none";
    rightBtn.style.display = "none";
    return;
  }

  // left button
  if (slider.scrollLeft <= 5) {
    leftBtn.style.display = "none";
  } else {
    leftBtn.style.display = "block";
  }

  // right button
  if (slider.scrollLeft + slider.clientWidth >= slider.scrollWidth - 5) {
    rightBtn.style.display = "none";
  } else {
    rightBtn.style.display = "block";
  }
}

function scrollLeft() {
  const slider = document.getElementById("recSlider");
  if (!slider) return;
  const newScroll = Math.max(slider.scrollLeft - REC_SCROLL_AMOUNT, 0);
  slider.scrollTo({ left: newScroll, behavior: "smooth" });
  // update buttons after scroll completes
  window.clearTimeout(slider._updateTimeout);
  slider._updateTimeout = window.setTimeout(updateScrollButtons, REC_SCROLL_DEBOUNCE);
}

function scrollRight() {
  const slider = document.getElementById("recSlider");
  if (!slider) return;
  const maxScroll = slider.scrollWidth - slider.clientWidth;
  const newScroll = Math.min(slider.scrollLeft + REC_SCROLL_AMOUNT, maxScroll);
  slider.scrollTo({ left: newScroll, behavior: "smooth" });
  window.clearTimeout(slider._updateTimeout);
  slider._updateTimeout = window.setTimeout(updateScrollButtons, REC_SCROLL_DEBOUNCE);
}

// Auto-scroll helpers
function startAutoScroll() {
  const slider = document.getElementById("recSlider");
  if (!slider) return;
  // don't auto-scroll if content doesn't overflow
  if (slider.scrollWidth <= slider.clientWidth + 2) return;

  stopAutoScroll(); // clear any existing
  recAutoInterval = setInterval(() => {
    // if at end -> go back to start smoothly
    if (slider.scrollLeft + slider.clientWidth >= slider.scrollWidth - 5) {
      slider.scrollTo({ left: 0, behavior: "smooth" });
    } else {
      slider.scrollBy({ left: REC_SCROLL_AMOUNT, behavior: "smooth" });
    }
    // update buttons after movement
    window.clearTimeout(slider._updateTimeout);
    slider._updateTimeout = window.setTimeout(updateScrollButtons, REC_SCROLL_DEBOUNCE + 100);
  }, 4000);
}

function stopAutoScroll() {
  if (recAutoInterval) {
    clearInterval(recAutoInterval);
    recAutoInterval = null;
  }
}

// When user clicks a recommendation card, show similar items
// Delegate click to container (safer than inline onclick)
document.addEventListener("click", function (e) {
  const card = e.target.closest && e.target.closest(".rec-card");
  if (card) {
    const name = card.getAttribute("data-name");
    if (name) showRecommendations(name);
  }
});

// =================== Show ML Recommendations in Modal ===================
function showRecommendations(name) {
  fetch(`/api/recommend?name=${encodeURIComponent(name)}`)
    .then(res => res.json())
    .then(data => {
      const modal = document.getElementById("recModal");
      const modalTitle = document.getElementById("modalTitle");
      const modalResults = document.getElementById("modalResults");

      if (!modal || !modalTitle || !modalResults) return;
      modalTitle.innerText = `Similar to ${name}`;
      modalResults.innerHTML = "";

      if (!data || data.length === 0) {
        modalResults.innerHTML = "<p>No similar restaurants found.</p>";
      } else {
        data.forEach(r => {
          const image = getRestaurantImage(r);
          modalResults.innerHTML += `
            <div class="card">
              <img src="${image}" alt="${escapeHtml(r["Restaurant Name"] || "Restaurant image")}" loading="lazy" onerror="this.onerror=null;this.src='/static/images/restaurant-default.jpg';">
              <div class="card-content">
                <h4>${escapeHtml(r["Restaurant Name"] || "")}</h4>
                <p><b>City:</b> ${escapeHtml(r.City || "")}</p>
                <p><b>Cuisine:</b> ${escapeHtml(r.Cuisines || "")}</p>
                <p><b>Rating:</b> ⭐ ${escapeHtml(r["Aggregate rating"] || "")}</p>
              </div>
            </div>`;
        });
      }
      modal.style.display = "block";
    })
    .catch(err => {
      console.error("Error fetching recommendations:", err);
    });
}

// Close modal utilities
function closeModal() {
  const modal = document.getElementById("recModal");
  if (modal) modal.style.display = "none";
}
window.onclick = function (event) {
  const modal = document.getElementById("recModal");
  if (event.target === modal) modal.style.display = "none";
};

// =================== Filters loading (no Choices.js) ===================
function loadFilters() {
  fetch("/api/filters")
    .then(res => res.json())
    .then(data => {
      console.log("Filters from backend:", data);
      const citySelect = document.getElementById("cityFilter");
      const cuisineSelect = document.getElementById("cuisineFilter");

      if (!citySelect || !cuisineSelect) return;

      citySelect.innerHTML = '<option value="">Select City</option>';
      cuisineSelect.innerHTML = '<option value="">Select Cuisine</option>';

      (data.cities || []).forEach(c => {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        citySelect.appendChild(opt);
      });

      (data.cuisines || []).forEach(c => {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        cuisineSelect.appendChild(opt);
      });
    })
    .catch(err => {
      console.error("Error loading filters:", err);
    });
}

// =================== On page load ===================
window.onload = function () {
  loadFilters();
  loadRestaurants();
  loadRecommendations();
  renderWishlist();
  // Smooth behavior for restaurant grid
  const grid = document.getElementById('restaurantGrid');
  if (grid) grid.style.scrollBehavior = 'smooth';

  // Character counters
  const msg = document.getElementById('message');
  const msgCount = document.getElementById('msgCount');
  if (msg && msgCount) {
    const update = () => { msgCount.textContent = String(msg.value.length); };
    msg.addEventListener('input', update);
    update();
  }
  const fbMsg = document.getElementById('fb_message');
  const fbMsgCount = document.getElementById('fbMsgCount');
  if (fbMsg && fbMsgCount) {
    const update2 = () => { fbMsgCount.textContent = String(fbMsg.value.length); };
    fbMsg.addEventListener('input', update2);
    update2();
  }

  // Predictive typing suggestion below search input
  const input = document.getElementById('search') || document.getElementById('search-input');
  const suggest = document.getElementById('searchSuggestion');
  if (input && suggest) {
    let timer = null;
    input.addEventListener('input', () => {
      const text = input.value.trim();
      if (!text) { suggest.style.display = 'none'; suggest.textContent=''; return; }
      window.clearTimeout(timer);
      timer = window.setTimeout(async () => {
        try {
          const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
          const data = await res.json();
          const token = data && (data.suggestion || data.next_token) ? String(data.suggestion || data.next_token) : '';
          if (token) {
            suggest.textContent = token;
            suggest.style.display = 'block';
          } else {
            suggest.style.display = 'none';
          }
        } catch (_) {
          suggest.style.display = 'none';
        }
      }, 250);
    });
  }
};

// 🌙 Dark Mode Toggle (unchanged)
document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("darkModeToggle");
  if (!toggle) return;
  const body = document.body;

  if (localStorage.getItem("darkMode") === "enabled") {
    body.classList.add("dark-mode");
    toggle.checked = true;
  }

  toggle.addEventListener("change", () => {
    if (toggle.checked) {
      body.classList.add("dark-mode");
      localStorage.setItem("darkMode", "enabled");
    } else {
      body.classList.remove("dark-mode");
      localStorage.setItem("darkMode", "disabled");
    }
  });
});

// Wishlist heart styles and interactions
const style = document.createElement('style');
style.textContent = `
.wishlist-icon { position:absolute; top:10px; right:10px; font-size:20px; cursor:pointer; user-select:none; transition: transform .15s ease, filter .2s ease; }
.wishlist-icon:hover { transform: scale(1.08); filter: drop-shadow(0 2px 4px rgba(0,0,0,.2)); }
.wishlist-icon.liked { filter: hue-rotate(10deg) saturate(1.2); }
`;
document.head.appendChild(style);

function toggleHeart(el, name){
  const liked = el.classList.toggle('liked');
  el.classList.toggle('active', liked);
  el.setAttribute('data-liked', liked ? 'true' : 'false');
  if (liked) {
    fetch('/api/wishlist', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name }) })
      .catch(()=>{});
  } else {
    fetch(`/api/wishlist/${encodeURIComponent(name)}`, { method:'DELETE' })
      .catch(()=>{});
  }
}

// Wishlist page rendering (full details)
function loadWishlistPage(){
  const grid = document.getElementById('wishlistGrid');
  if (!grid) return;
  fetch('/api/wishlist/details')
    .then(r=>r.json())
    .then(items => {
      grid.innerHTML = '';
      if (!items || items.length === 0) { grid.innerHTML = '<p>No liked restaurants yet.</p>'; return; }
      items.forEach(r => {
        const image = getRestaurantImage(r);
        const html = `
          <div class="card" style="position:relative;">
            <span class="wishlist-icon liked" onclick="toggleHeart(this, '${escapeJsString(r["Restaurant Name"] || "")}')">❤️</span>
            <img src="${image}" alt="${escapeHtml(r["Restaurant Name"] || "Restaurant image")}" loading="lazy" onerror="this.onerror=null;this.src='/static/images/restaurant-default.jpg';">
            <div class="card-content">
              <h3>${escapeHtml(r["Restaurant Name"] || "")}</h3>
              <p><b>City:</b> ${escapeHtml(r.City || "")}</p>
              <p><b>Cuisine:</b> ${escapeHtml(r.Cuisines || "")}</p>
              <p><b>Rating:</b> ⭐ ${escapeHtml(r["Aggregate rating"] || "")}</p>
              <form method="post" action="/wishlist/remove">
                <input type="hidden" name="item_id" value="${escapeHtml(r["Restaurant Name"] || "")}">
                <button type="submit" class="remove-btn">Remove</button>
              </form>
            </div>
          </div>`;
        grid.insertAdjacentHTML('beforeend', html);
      });
    })
    .catch(()=>{ grid.innerHTML = '<p>Error loading wishlist.</p>'; });
}

// Auto-load wishlist page if present
document.addEventListener('DOMContentLoaded', loadWishlistPage);

// =============== Client-side form validation ===============
document.addEventListener("submit", function (e) {
  const form = e.target;
  if (!(form instanceof HTMLFormElement)) return;
  const requiredInputs = form.querySelectorAll("input[required], textarea[required], select[required]");
  let valid = true;
  requiredInputs.forEach((el) => {
    if (!el.value || (el.type === "email" && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(el.value))) {
      valid = false;
      el.style.borderColor = "#b91c1c"; // subtle error
      el.style.boxShadow = "0 0 0 3px rgba(185,28,28,0.15)";
    } else {
      el.style.borderColor = "";
      el.style.boxShadow = "";
    }
  });
  if (!valid) {
    e.preventDefault();
  }
});

// =============== Subtle hover lift for .card buttons ===============
document.addEventListener("mouseover", function (e) {
  const btn = e.target.closest && e.target.closest(".card button");
  if (btn) btn.style.transform = "translateY(-1px)";
});
document.addEventListener("mouseout", function (e) {
  const btn = e.target.closest && e.target.closest(".card button");
  if (btn) btn.style.transform = "";
});