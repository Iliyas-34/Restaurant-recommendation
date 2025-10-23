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

// =============== Predictive Typing Functionality ===============
let searchTimeout;
let currentSuggestions = [];

// Enhanced search input handler with predictive typing
function handleSearchInput() {
    const searchInput = document.getElementById('search-input');
    const query = searchInput.value.trim();
    
    // Clear previous timeout
    if (searchTimeout) {
        clearTimeout(searchTimeout);
    }
    
    // If query is too short, clear suggestions
    if (query.length < 2) {
        hideSearchSuggestions();
        return;
    }
    
    // Debounce the search
    searchTimeout = setTimeout(async () => {
        try {
            // Get suggestions from the new API
            const response = await fetch(`/api/suggestions?q=${encodeURIComponent(query)}&limit=8`);
            const data = await response.json();
            
            if (data.suggestions && data.suggestions.length > 0) {
                displaySearchSuggestions(data.suggestions);
            } else {
                hideSearchSuggestions();
            }
        } catch (error) {
            console.error('Error getting suggestions:', error);
            hideSearchSuggestions();
        }
    }, 300);
}

// Display search suggestions
function displaySearchSuggestions(suggestions) {
    const searchWrapper = document.querySelector('.search-wrapper');
    let suggestionsContainer = document.getElementById('search-suggestions');
    
    if (!suggestionsContainer) {
        suggestionsContainer = document.createElement('div');
        suggestionsContainer.id = 'search-suggestions';
        suggestionsContainer.className = 'search-suggestions';
        searchWrapper.appendChild(suggestionsContainer);
    }
    
    suggestionsContainer.innerHTML = suggestions.map(suggestion => {
        const text = suggestion.text || suggestion;
        const type = suggestion.type || 'restaurant';
        const city = suggestion.city || '';
        const cuisines = suggestion.cuisines || '';
        
        let icon = '🍽️';
        if (type === 'cuisine') icon = '🍴';
        else if (type === 'city') icon = '📍';
        
        return `
            <div class="suggestion-item" onclick="selectSuggestion('${text.replace(/'/g, "\\'")}')">
                <span class="suggestion-icon">${icon}</span>
                <span class="suggestion-text">${text}</span>
                ${city ? `<span class="suggestion-meta">${city}</span>` : ''}
                ${cuisines ? `<span class="suggestion-meta">${cuisines}</span>` : ''}
            </div>
        `;
    }).join('');
    
    suggestionsContainer.style.display = 'block';
}

// Hide search suggestions
function hideSearchSuggestions() {
    const suggestionsContainer = document.getElementById('search-suggestions');
    if (suggestionsContainer) {
        suggestionsContainer.style.display = 'none';
    }
}

// Select a suggestion
function selectSuggestion(suggestion) {
    const searchInput = document.getElementById('search-input');
    searchInput.value = suggestion;
    hideSearchSuggestions();
    applySmartFilters();
}

// =============== Smart Filters Functionality ===============
async function applySmartFilters() {
  const mood = document.getElementById('mood-filter').value;
  const time = document.getElementById('time-filter').value;
  const occasion = document.getElementById('occasion-filter').value;
  const searchInput = document.getElementById('search-input').value.trim();
  
  // Show loading state
  const filterBtn = document.querySelector('.primary-filter-btn');
  const originalText = filterBtn.innerHTML;
  filterBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Finding Perfect Match...';
  filterBtn.disabled = true;
  
  try {
    // Use the new search API with filters
    const searchParams = new URLSearchParams({
      q: searchInput || "restaurant",
      mood: mood,
      time: time,
      occasion: occasion,
      per_page: 20
    });
    
    console.log('Applying smart filters:', { mood, time, occasion, searchInput });
    
    const response = await fetch(`/api/search?${searchParams}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Smart filter results:', data);
    
    if (data.error) {
      throw new Error(data.error);
    }
    
    // Display the results
    displaySmartFilterResults(data.restaurants, {
      user_input: searchInput,
      mood: mood,
      time: time,
      occasion: occasion
    });
    
  } catch (error) {
    console.error('Error applying smart filters:', error);
    displayError(`Failed to load restaurant recommendations. Please check your connection and try again.`);
  } finally {
    // Restore button state
    filterBtn.innerHTML = originalText;
    filterBtn.disabled = false;
  }
}

function clearAllFilters() {
  // Clear all filter selects
  document.getElementById('mood-filter').value = '';
  document.getElementById('time-filter').value = '';
  document.getElementById('occasion-filter').value = '';
  document.getElementById('search-input').value = '';
  
  // Hide any existing results
  const resultsContainer = document.getElementById('smart-filter-results');
  if (resultsContainer) {
    resultsContainer.style.display = 'none';
  }
  
  // Show success message
  const filterBtn = document.querySelector('.secondary-filter-btn');
  const originalText = filterBtn.innerHTML;
  filterBtn.innerHTML = '<i class="fa fa-check"></i> Cleared!';
  filterBtn.style.background = '#10B981';
  filterBtn.style.color = 'white';
  
  setTimeout(() => {
    filterBtn.innerHTML = originalText;
    filterBtn.style.background = '';
    filterBtn.style.color = '';
  }, 1500);
}

function displaySmartFilterResults(restaurants, query) {
  // Create or update results container
  let resultsContainer = document.getElementById('smart-filter-results');
  if (!resultsContainer) {
    resultsContainer = document.createElement('div');
    resultsContainer.id = 'smart-filter-results';
    resultsContainer.className = 'smart-filter-results';
    document.querySelector('.hero').appendChild(resultsContainer);
  }
  
  if (!restaurants || restaurants.length === 0) {
    resultsContainer.innerHTML = `
      <div class="no-results">
        <h3>No perfect matches found</h3>
        <p>Try adjusting your filters or search terms.</p>
      </div>
    `;
    resultsContainer.style.display = 'block';
    return;
  }
  
  // Build results HTML
  const queryText = `for ${query.user_input || 'restaurants'} (${query.mood || 'any'} mood, ${query.time || 'any time'}, ${query.occasion || 'any occasion'})`;
  
  resultsContainer.innerHTML = `
    <div class="results-header">
      <h3>🎯 Perfect Matches ${queryText}</h3>
      <p>Based on your preferences and mood</p>
    </div>
    <div class="results-grid">
      ${restaurants.map(restaurant => `
        <div class="smart-result-card" onclick="goToRestaurant('${restaurant.name}')">
          <div class="result-image">
            <img src="${getCuisineImage(restaurant.cuisines)}" alt="${restaurant.name}" onerror="this.src='/static/images/restaurant-default.jpg'">
            <div class="rating-badge">⭐ ${restaurant.rating.toFixed(1)}</div>
          </div>
          <div class="result-info">
            <h4>${restaurant.name}</h4>
            <p class="cuisine">${restaurant.cuisines}</p>
            <p class="location">📍 ${restaurant.address || restaurant.location}</p>
            <div class="result-meta">
              <span class="rating">⭐ ${restaurant.rating.toFixed(1)}/5</span>
              <span class="price">💰 ${restaurant.price_range}</span>
            </div>
            <div class="result-actions">
              <button class="view-btn" onclick="event.stopPropagation(); goToRestaurant('${restaurant.name}')">View Details</button>
              <button class="wishlist-btn" onclick="event.stopPropagation(); addToWishlist('${restaurant.name}')">❤️</button>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
  
  resultsContainer.style.display = 'block';
  
  // Scroll to results
  resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Add CSS for smart filter results
const smartFilterCSS = `
<style>
.smart-filter-results {
  margin-top: 40px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  max-width: 1000px;
  margin-left: auto;
  margin-right: auto;
}

.results-header {
  text-align: center;
  margin-bottom: 30px;
}

.results-header h3 {
  color: #1F2933;
  font-size: 1.5rem;
  margin-bottom: 10px;
}

.results-header p {
  color: #6B7280;
  font-size: 1rem;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

        .smart-result-card {
          background: white;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
          transition: transform 0.3s ease;
          cursor: pointer;
        }

        .smart-result-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }

.result-image {
  position: relative;
  height: 150px;
  overflow: hidden;
}

.result-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.match-score {
  position: absolute;
  top: 10px;
  right: 10px;
  background: #D9822B;
  color: white;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: bold;
}

.result-info {
  padding: 15px;
}

.result-info h4 {
  color: #1F2933;
  font-size: 1.1rem;
  margin-bottom: 5px;
}

.result-info .cuisine {
  color: #D9822B;
  font-size: 0.9rem;
  margin-bottom: 5px;
}

.result-info .location {
  color: #6B7280;
  font-size: 0.9rem;
  margin-bottom: 10px;
}

.result-meta {
  display: flex;
  justify-content: space-between;
  margin-bottom: 15px;
}

.result-meta span {
  font-size: 0.9rem;
  color: #374151;
}

.view-btn {
  width: 100%;
  background: #D9822B;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.3s ease;
}

        .view-btn:hover {
          background: #B8651F;
        }

        .result-actions {
          display: flex;
          gap: 8px;
          margin-top: 10px;
        }

        .wishlist-btn {
          background: #EF4444;
          color: white;
          border: none;
          padding: 8px 12px;
          border-radius: 6px;
          cursor: pointer;
          transition: background 0.3s ease;
          font-size: 14px;
        }

        .wishlist-btn:hover {
          background: #DC2626;
        }

        .rating-badge {
          position: absolute;
          top: 10px;
          right: 10px;
          background: #D9822B;
          color: white;
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: bold;
        }

        .no-results {
          text-align: center;
          padding: 40px;
          color: #6B7280;
        }

        .error-message {
          text-align: center;
          padding: 40px;
          color: #EF4444;
        }

        .suggestion-item {
          padding: 12px 16px;
          border-bottom: 1px solid #E5E7EB;
          cursor: pointer;
          transition: background 0.2s ease;
        }

        .suggestion-item:hover {
          background: #F9FAFB;
        }

        .suggestion-item:last-child {
          border-bottom: none;
        }

        .suggestion-details {
          color: #6B7280;
          font-size: 14px;
          margin-left: 8px;
        }

@media (max-width: 768px) {
  .results-grid {
    grid-template-columns: 1fr;
  }
}
</style>
`;

// Inject CSS
document.head.insertAdjacentHTML('beforeend', smartFilterCSS);

// =============== Restaurant Navigation ===============
function goToRestaurant(restaurantName) {
  // Navigate to restaurant details page
  window.location.href = `/restaurant/${encodeURIComponent(restaurantName)}`;
}

// =============== Restaurant Details Functions ===============
function toggleWishlist(restaurantName) {
  // Use the same logic as toggleHeart but for the details page
  const btn = document.querySelector('.btn-wishlist');
  if (!btn) return;
  
  const isLiked = btn.classList.contains('liked');
  
  if (isLiked) {
    // Remove from wishlist
    fetch(`/api/wishlist/${encodeURIComponent(restaurantName)}`, { method: 'DELETE' })
      .then(() => {
        btn.classList.remove('liked');
        btn.innerHTML = '<i class="fa fa-heart"></i> Add to Wishlist';
        showNotification('Removed from wishlist', 'info');
      })
      .catch(() => {
        showNotification('Failed to remove from wishlist', 'error');
      });
  } else {
    // Add to wishlist
    fetch('/api/wishlist', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: restaurantName })
    })
      .then(() => {
        btn.classList.add('liked');
        btn.innerHTML = '<i class="fa fa-heart"></i> Remove from Wishlist';
        showNotification('Added to wishlist!', 'success');
      })
      .catch(() => {
        showNotification('Failed to add to wishlist', 'error');
      });
  }
}

function shareRestaurant() {
  if (navigator.share) {
    navigator.share({
      title: document.querySelector('.restaurant-name')?.textContent || 'Restaurant',
      text: 'Check out this restaurant!',
      url: window.location.href
    });
  } else {
    // Fallback: copy to clipboard
    navigator.clipboard.writeText(window.location.href).then(() => {
      showNotification('Link copied to clipboard!', 'success');
    }).catch(() => {
      showNotification('Failed to copy link', 'error');
    });
  }
}

// =============== Wishlist Functions ===============
async function addToWishlist(restaurantName) {
  try {
    const response = await fetch('/api/wishlist', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name: restaurantName })
    });
    
    if (response.ok) {
      // Show success message
      showNotification('Added to wishlist!', 'success');
    } else {
      throw new Error('Failed to add to wishlist');
    }
  } catch (error) {
    console.error('Error adding to wishlist:', error);
    showNotification('Failed to add to wishlist', 'error');
  }
}

function showNotification(message, type = 'info') {
  // Create notification element
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.textContent = message;
  
  // Add styles
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 12px 20px;
    border-radius: 6px;
    color: white;
    font-weight: 500;
    z-index: 1000;
    animation: slideIn 0.3s ease;
    background: ${type === 'success' ? '#10B981' : type === 'error' ? '#EF4444' : '#3B82F6'};
  `;
  
  document.body.appendChild(notification);
  
  // Remove after 3 seconds
  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 300);
  }, 3000);
}

// =============== Search and Filter Functionality ===============
// Enhanced search and filter functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const moodFilter = document.getElementById('mood-filter');
    const timeFilter = document.getElementById('time-filter');
    const occasionFilter = document.getElementById('occasion-filter');
    
    // Load featured restaurants on page load
    loadFeaturedRestaurants();
    
    // Initialize filter listeners for immediate updates
    initializeFilterListeners();
    
    // Real-time search suggestions
    if (searchInput) {
        let searchTimeout;
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                updateSearchSuggestions(this.value);
            }, 300);
        });
    }
    
    // Filter change handlers
    [moodFilter, timeFilter, occasionFilter].forEach(filter => {
        if (filter) {
            filter.addEventListener('change', function() {
                updateFilteredResults();
            });
        }
    });
});

async function updateSearchSuggestions(query) {
    // Implement search suggestions based on Zomato data
    if (query.length < 2) return;
    
    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&per_page=5`);
        const data = await response.json();
        
        if (data.restaurants && data.restaurants.length > 0) {
            // Extract unique suggestions from restaurant names and cuisines
            const suggestions = new Set();
            data.restaurants.forEach(restaurant => {
                if (restaurant.name && restaurant.name.toLowerCase().includes(query.toLowerCase())) {
                    suggestions.add(restaurant.name);
                }
                if (restaurant.cuisines) {
                    restaurant.cuisines.split(',').forEach(cuisine => {
                        const trimmedCuisine = cuisine.trim();
                        if (trimmedCuisine.toLowerCase().includes(query.toLowerCase())) {
                            suggestions.add(trimmedCuisine);
                        }
                    });
                }
            });
            
            if (suggestions.size > 0) {
                displaySearchSuggestions(Array.from(suggestions).slice(0, 5));
            }
        }
    } catch (error) {
        console.error('Error fetching search suggestions:', error);
    }
}

function displaySearchSuggestions(suggestions) {
    const suggestionBox = document.getElementById('suggestion-box');
    if (!suggestionBox) return;
    
    let suggestionsHtml = '';
    
    if (suggestions.length > 0) {
        // Check if suggestions are restaurant objects or strings
        if (typeof suggestions[0] === 'object' && suggestions[0].name) {
            // Restaurant objects
            suggestionsHtml = suggestions.map(restaurant => `
                <div class="suggestion-item" onclick="selectSuggestion('${restaurant.name}')">
                    <strong>${restaurant.name}</strong>
                    <span class="suggestion-details">${restaurant.cuisines} • ${restaurant.city}</span>
                </div>
            `).join('');
        } else {
            // String suggestions
            suggestionsHtml = suggestions.map(suggestion => `
                <div class="suggestion-item" onclick="selectSuggestion('${suggestion}')">
                    <strong>${suggestion}</strong>
                </div>
            `).join('');
        }
    }
    
    suggestionBox.innerHTML = suggestionsHtml;
    suggestionBox.style.display = suggestionsHtml ? 'block' : 'none';
}

function selectSuggestion(restaurantName) {
    document.getElementById('search-input').value = restaurantName;
    document.getElementById('suggestion-box').style.display = 'none';
    updateFilteredResults();
}

async function updateFilteredResults() {
    // Implement dynamic filtering based on current selections
    const mood = document.getElementById('mood-filter')?.value || '';
    const time = document.getElementById('time-filter')?.value || '';
    const occasion = document.getElementById('occasion-filter')?.value || '';
    const searchQuery = document.getElementById('search-input')?.value || '';
    
    try {
        const searchParams = new URLSearchParams({
            q: searchQuery,
            mood: mood,
            time: time,
            occasion: occasion,
            per_page: 20
        });
        
        const response = await fetch(`/api/search?${searchParams}`);
        const data = await response.json();
        
        if (data.restaurants && data.restaurants.length > 0) {
            displaySmartFilterResults(data.restaurants, {
                user_input: searchQuery,
                mood: mood,
                time: time,
                occasion: occasion
            });
        } else {
            displayNoResults();
        }
    } catch (error) {
        console.error('Error updating filtered results:', error);
        displayError('Failed to load results. Please try again.');
    }
}

function displayNoResults() {
    const resultsContainer = document.getElementById('smart-filter-results');
    if (!resultsContainer) return;
    
    resultsContainer.innerHTML = `
        <div class="no-results">
            <h3>No restaurants found</h3>
            <p>Try adjusting your search terms or filters.</p>
        </div>
    `;
    resultsContainer.style.display = 'block';
}

function displayError(message, action = null) {
    const resultsContainer = document.getElementById('smart-filter-results');
    if (!resultsContainer) return;
    
    const retryButton = action ? `<button class="retry-btn" onclick="${action}">Try Again</button>` : '';
    resultsContainer.innerHTML = `
        <div class="error-message">
            <div class="error-icon">⚠️</div>
            <h3>Oops! Something went wrong</h3>
            <p>${message}</p>
            ${retryButton}
        </div>
    `;
    resultsContainer.style.display = 'block';
}

// =============== Filter Event Listeners ===============
function initializeFilterListeners() {
    // Add event listeners to all filter inputs
    const moodFilter = document.getElementById('mood-filter');
    const timeFilter = document.getElementById('time-filter');
    const occasionFilter = document.getElementById('occasion-filter');
    
    if (moodFilter) {
        moodFilter.addEventListener('change', applySmartFilters);
    }
    if (timeFilter) {
        timeFilter.addEventListener('change', applySmartFilters);
    }
    if (occasionFilter) {
        occasionFilter.addEventListener('change', applySmartFilters);
    }
}

// =============== Featured Restaurants ===============
async function loadFeaturedRestaurants() {
    try {
        const response = await fetch('/api/featured');
        const data = await response.json();
        
        if (data.restaurants && data.restaurants.length > 0) {
            displayFeaturedRestaurants(data.restaurants);
        } else {
            displayFeaturedError('No featured restaurants available');
        }
    } catch (error) {
        console.error('Error loading featured restaurants:', error);
        displayFeaturedError('Failed to load featured restaurants');
    }
}

function displayFeaturedRestaurants(restaurants) {
    const container = document.getElementById('featuredRestaurants');
    if (!container) return;
    
    container.innerHTML = restaurants.map(restaurant => `
        <div class="restaurant-card featured">
            <div class="restaurant-image">
                <img src="${getCuisineImage(restaurant.cuisines)}" alt="${restaurant.name}" onerror="this.src='/static/images/food1.jpg'">
                <div class="rating-badge">${restaurant.rating.toFixed(1)} ⭐</div>
            </div>
            <div class="restaurant-info">
                <h3>${restaurant.name}</h3>
                <p class="cuisine-tag">${restaurant.cuisines}</p>
                <p class="restaurant-description">${restaurant.location || restaurant.city || 'Great dining experience'}</p>
                <div class="restaurant-meta">
                    <span class="price-range">₹${restaurant.price}</span>
                    <span class="distance">${restaurant.votes} votes</span>
                </div>
                <button class="view-details-btn" onclick="goToRestaurant('${restaurant.name}')">View Details</button>
            </div>
        </div>
    `).join('');
}

function displayFeaturedError(message) {
    const container = document.getElementById('featuredRestaurants');
    if (!container) return;
    
    container.innerHTML = `
        <div class="error-message">
            <p>${message}</p>
        </div>
    `;
}

// =============== Nearby Restaurants ===============
function getCurrentLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                searchNearbyRestaurants(lat, lng);
            },
            function(error) {
                console.error('Error getting location:', error);
                alert('Unable to get your location. Please enter it manually.');
            }
        );
    } else {
        alert('Geolocation is not supported by this browser.');
    }
}

async function searchNearbyRestaurants(lat = null, lng = null) {
    const locationInput = document.getElementById('location-input');
    const container = document.getElementById('nearbyRestaurants');
    
    if (!container) return;
    
    // Show loading state
    container.innerHTML = '<div class="loading-placeholder"><p>Searching nearby restaurants...</p></div>';
    
    try {
        let url = '/api/nearby?';
        
        if (lat && lng) {
            url += `lat=${lat}&lng=${lng}`;
        } else if (locationInput && locationInput.value.trim()) {
            url += `city=${encodeURIComponent(locationInput.value.trim())}`;
        } else {
            container.innerHTML = '<div class="loading-placeholder"><p>Please enter a location or use current location</p></div>';
            return;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.restaurants && data.restaurants.length > 0) {
            displayNearbyRestaurants(data.restaurants);
        } else {
            displayNearbyError('No restaurants found nearby. Try a different location.');
        }
    } catch (error) {
        console.error('Error searching nearby restaurants:', error);
        displayNearbyError('Failed to search nearby restaurants');
    }
}

function displayNearbyRestaurants(restaurants) {
    const container = document.getElementById('nearbyRestaurants');
    if (!container) return;
    
    container.innerHTML = restaurants.map(restaurant => `
        <div class="nearby-card">
            <img src="${getCuisineImage(restaurant.cuisines)}" alt="${restaurant.name}" onerror="this.src='/static/images/food1.jpg'">
            <div class="nearby-info">
                <h4>${restaurant.name}</h4>
                <p>${restaurant.distance ? restaurant.distance.toFixed(1) + ' km' : 'Nearby'} • ${restaurant.cuisines}</p>
                <div class="quick-actions">
                    <button class="quick-btn" onclick="goToRestaurant('${restaurant.name}')">View Details</button>
                    <button class="quick-btn" onclick="openDirections(${restaurant.latitude}, ${restaurant.longitude})">Directions</button>
                </div>
            </div>
        </div>
    `).join('');
}

function displayNearbyError(message) {
    const container = document.getElementById('nearbyRestaurants');
    if (!container) return;
    
    container.innerHTML = `
        <div class="error-message">
            <p>${message}</p>
        </div>
    `;
}

function openDirections(lat, lng) {
    if (lat && lng) {
        const url = `https://www.google.com/maps?q=${lat},${lng}`;
        window.open(url, '_blank');
    } else {
        alert('Location coordinates not available');
    }
}