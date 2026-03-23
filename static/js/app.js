(function () {
  var mode = "single";

  var uploadBox = document.getElementById("uploadBox");
  var imageInput = document.getElementById("imageInput");
  var previewImage = document.getElementById("previewImage");
  var previewPlaceholder = document.getElementById("previewPlaceholder");
  var searchForm = document.getElementById("searchForm");
  var runSearchBtn = document.getElementById("runSearchBtn");
  var modeSingle = document.getElementById("modeSingle");
  var modeOutfit = document.getElementById("modeOutfit");
  var statusBox = document.getElementById("statusBox");
  var inputModeUpload = document.getElementById("inputModeUpload");
  var inputModeUrl = document.getElementById("inputModeUrl");
  var imageUrlGroup = document.getElementById("imageUrlGroup");
  var imageUrlInput = document.getElementById("imageUrlInput");

  var minSimilarity = document.getElementById("minSimilarity");
  var maxResults = document.getElementById("maxResults");

  var singleResultsGrid = document.getElementById("singleResultsGrid");
  var emptySingleResults = document.getElementById("emptySingleResults");

  var outfitResults = document.getElementById("outfitResults");
  var outfitSummary = document.getElementById("outfitSummary");
  var detectedItems = document.getElementById("detectedItems");
  var outfitRecommendations = document.getElementById("outfitRecommendations");

  if (!uploadBox || !imageInput || !searchForm) {
    return;
  }

  var inputMode = "upload";
  var urlPreviewTimer = null;
  var previewedUrl = "";

  function setStatus(msg, isError) {
    if (!statusBox) return;
    statusBox.textContent = msg || "";
    statusBox.style.color = isError ? "#b42318" : "#5b6475";
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function toDataUrl(base64Str) {
    if (!base64Str) return "";
    if (base64Str.indexOf("data:image") === 0) return base64Str;
    return "data:image/jpeg;base64," + base64Str;
  }

  function updateMode(nextMode) {
    mode = nextMode;
    if (modeSingle && modeOutfit) {
      modeSingle.classList.toggle("is-active", mode === "single");
      modeOutfit.classList.toggle("is-active", mode === "outfit");
    }
    if (runSearchBtn) {
      runSearchBtn.textContent = mode === "single" ? "Find Similar Products" : "Analyze Outfit";
    }
    clearResults();
    setStatus("");
  }

  function updateInputMode(nextInputMode) {
    inputMode = nextInputMode;
    if (inputModeUpload && inputModeUrl) {
      inputModeUpload.classList.toggle("is-active", inputMode === "upload");
      inputModeUrl.classList.toggle("is-active", inputMode === "url");
    }
    if (imageUrlGroup) {
      imageUrlGroup.classList.toggle("is-hidden", inputMode !== "url");
    }
    if (uploadBox) {
      uploadBox.classList.toggle("is-hidden", inputMode !== "upload");
    }

    // Prevent browser native validation from blocking submit in URL mode.
    if (imageInput) {
      imageInput.required = inputMode === "upload";
    }
    if (imageUrlInput) {
      imageUrlInput.required = inputMode === "url";
    }

    if (inputMode === "url") {
      triggerUrlPreview(true);
    } else {
      var file = imageInput.files && imageInput.files[0];
      if (file) {
        renderPreview(file);
      } else {
        clearPreview();
      }
      setStatus("");
    }
  }

  function ensurePreviewImage() {
    if (!previewImage) {
      var frame = document.querySelector(".preview-frame");
      if (frame) {
        previewImage = document.createElement("img");
        previewImage.id = "previewImage";
        previewImage.alt = "Input image preview";
        frame.insertBefore(previewImage, frame.firstChild);
      }
    }
    return previewImage;
  }

  function clearPreview() {
    var img = ensurePreviewImage();
    if (img) {
      img.src = "";
      img.classList.add("is-hidden");
    }
    if (previewPlaceholder) previewPlaceholder.classList.remove("is-hidden");
  }

  function isValidHttpUrl(url) {
    try {
      var parsed = new URL(url);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch (e) {
      return false;
    }
  }

  function renderPreviewFromUrl(url) {
    return new Promise(function (resolve, reject) {
      if (!isValidHttpUrl(url)) {
        clearPreview();
        previewedUrl = "";
        setStatus("Invalid image URL or unable to load image", true);
        reject(new Error("invalid-url"));
        return;
      }

      setStatus("Loading image preview...");
      var probe = new Image();
      probe.onload = function () {
        var img = ensurePreviewImage();
        if (!img) {
          setStatus("Invalid image URL or unable to load image", true);
          reject(new Error("no-preview-image"));
          return;
        }
        img.src = url;
        img.classList.remove("is-hidden");
        if (previewPlaceholder) previewPlaceholder.classList.add("is-hidden");
        previewedUrl = url;
        setStatus("Preview ready.");
        resolve(true);
      };
      probe.onerror = function () {
        clearPreview();
        previewedUrl = "";
        setStatus("Invalid image URL or unable to load image", true);
        reject(new Error("image-load-failed"));
      };
      probe.src = url;
    });
  }

  function triggerUrlPreview(immediate) {
    if (inputMode !== "url") return;
    if (!imageUrlInput) return;

    var url = (imageUrlInput.value || "").trim();

    if (urlPreviewTimer) {
      clearTimeout(urlPreviewTimer);
      urlPreviewTimer = null;
    }

    if (!url) {
      previewedUrl = "";
      clearPreview();
      setStatus("");
      return;
    }

    var delay = immediate ? 0 : 450;
    urlPreviewTimer = setTimeout(function () {
      renderPreviewFromUrl(url).then(function () {
        setStatus("Preview ready. Click Analyze.");
      }).catch(function () {
        // Status message already handled by renderPreviewFromUrl.
      });
    }, delay);
  }

  function clearResults() {
    if (singleResultsGrid) {
      var cards = singleResultsGrid.querySelectorAll(".product-card");
      cards.forEach(function (n) {
        n.remove();
      });
    }
    if (emptySingleResults) {
      emptySingleResults.classList.remove("is-hidden");
    }
    if (outfitResults) {
      outfitResults.classList.add("is-hidden");
    }
    if (outfitSummary) outfitSummary.innerHTML = "";
    if (detectedItems) detectedItems.innerHTML = "";
    if (outfitRecommendations) outfitRecommendations.innerHTML = "";
  }

  ["dragenter", "dragover"].forEach(function (evt) {
    uploadBox.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      uploadBox.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach(function (evt) {
    uploadBox.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      uploadBox.classList.remove("drag-active");
    });
  });

  uploadBox.addEventListener("drop", function (e) {
    if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files[0]) return;
    imageInput.files = e.dataTransfer.files;
    renderPreview(e.dataTransfer.files[0]);
  });

  imageInput.addEventListener("change", function (e) {
    if (!e.target.files || !e.target.files[0]) return;
    previewedUrl = "";
    renderPreview(e.target.files[0]);
  });

  if (imageUrlInput) {
    imageUrlInput.addEventListener("input", function () {
      // Keep flow explicit: paste URL -> press Enter -> preview loads.
      var url = (imageUrlInput.value || "").trim();
      if (!url) {
        previewedUrl = "";
        clearPreview();
        setStatus("");
        return;
      }
      if (url !== previewedUrl) {
        previewedUrl = "";
        clearPreview();
        setStatus("Press Enter to load preview.");
      }
    });

    imageUrlInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        triggerUrlPreview(true);
      }
    });
  }

  // Safety: do not submit the whole form when Enter is pressed in URL input.
  if (searchForm && imageUrlInput) {
    searchForm.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && e.target === imageUrlInput) {
        e.preventDefault();
      }
    });
  }

  if (modeSingle) {
    modeSingle.addEventListener("click", function () {
      updateMode("single");
    });
  }
  if (modeOutfit) {
    modeOutfit.addEventListener("click", function () {
      updateMode("outfit");
    });
  }

  if (inputModeUpload) {
    inputModeUpload.addEventListener("click", function () {
      updateInputMode("upload");
    });
  }
  if (inputModeUrl) {
    inputModeUrl.addEventListener("click", function () {
      updateInputMode("url");
    });
  }

  searchForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    var file = null;
    var url = "";
    var endpoint = "";
    if (inputMode === "upload") {
      file = imageInput.files && imageInput.files[0];
      if (!file) {
        setStatus("Please upload an image first.", true);
        return;
      }
      endpoint = mode === "single" ? "/recommend/" : "/outfit-analysis/";
    } else {
      url = imageUrlInput && imageUrlInput.value ? imageUrlInput.value.trim() : "";
      if (!url) {
        setStatus("Please paste an image URL.", true);
        return;
      }

      try {
        if (url !== previewedUrl) {
          setStatus("Press Enter to load preview first.", true);
          return;
        }
      } catch (previewErr) {
        return;
      }
      endpoint = mode === "single" ? "/recommend-url/" : "/outfit-analysis-url/";
    }

    setStatus(mode === "single" ? "Searching similar products..." : "Running full outfit analysis...");
    runSearchBtn.disabled = true;
    var oldBtnLabel = runSearchBtn.textContent;
    runSearchBtn.textContent = mode === "single" ? "Analyzing..." : "Running Analysis...";

    var formData = new FormData();
    if (inputMode === "upload") {
      formData.append("file", file);
    } else {
      formData.append("image_url", url);
    }

    try {
      if (mode === "single") {
        await runSingleSearch(formData, endpoint);
      } else {
        await runOutfitAnalysis(formData, endpoint);
      }
    } catch (err) {
      // URL-mode fallback: if URL endpoint is unavailable, try legacy file upload path.
      if (inputMode === "url") {
        try {
          setStatus("Primary URL route failed, trying fallback...");
          var fallbackData = await buildUploadFormDataFromUrl(url);
          if (mode === "single") {
            await runSingleSearch(fallbackData, "/recommend/");
          } else {
            await runOutfitAnalysis(fallbackData, "/outfit-analysis/");
          }
          return;
        } catch (fallbackErr) {
          var msg = fallbackErr && fallbackErr.message ? fallbackErr.message : "Request failed";
          setStatus(msg, true);
          return;
        }
      }
      setStatus(err.message || "Request failed", true);
    } finally {
      runSearchBtn.disabled = false;
      runSearchBtn.textContent = oldBtnLabel;
    }
  });

  async function readApiResponse(res) {
    var contentType = (res.headers.get("content-type") || "").toLowerCase();
    if (contentType.indexOf("application/json") !== -1) {
      return await res.json();
    }

    var raw = await res.text();
    try {
      return JSON.parse(raw);
    } catch (e) {
      return { error: raw || ("HTTP " + res.status) };
    }
  }

  async function buildUploadFormDataFromUrl(url) {
    if (!url) {
      throw new Error("Please paste an image URL.");
    }

    var resp = await fetch(url);
    if (!resp.ok) {
      throw new Error("Invalid image URL or unable to load image");
    }

    var blob = await resp.blob();
    var file = new File([blob], "url-image.jpg", { type: blob.type || "image/jpeg" });
    var fd = new FormData();
    fd.append("file", file);
    return fd;
  }

  async function runSingleSearch(formData, endpoint) {
    var res = await fetch(endpoint || "/recommend/", { method: "POST", body: formData });
    var data = await readApiResponse(res);

    if (!res.ok || data.error) {
      throw new Error(data.error || "Single search failed");
    }

    var products = Array.isArray(data.recommendations) ? data.recommendations : [];
    var minSim = parseFloat(minSimilarity && minSimilarity.value ? minSimilarity.value : "0");
    var maxK = parseInt(maxResults && maxResults.value ? maxResults.value : "8", 10);

    products = products
      .filter(function (p) {
        return Number(p.similarity_score || 0) >= minSim;
      })
      .slice(0, maxK);

    renderSingleResults(products);
    setStatus("Found " + products.length + " matches.");
  }

  async function runOutfitAnalysis(formData, endpoint) {
    var res = await fetch(endpoint || "/outfit-analysis/", { method: "POST", body: formData });
    var data = await readApiResponse(res);

    if (!res.ok || data.error) {
      throw new Error(data.error || "Outfit analysis failed");
    }

    renderOutfitResults(data);
    setStatus("Outfit analysis complete.");
  }

  function renderSingleResults(products) {
    clearResults();
    if (!singleResultsGrid) return;

    if (!products.length) {
      if (emptySingleResults) emptySingleResults.classList.remove("is-hidden");
      return;
    }

    if (emptySingleResults) emptySingleResults.classList.add("is-hidden");

    products.forEach(function (product) {
      var imageSrc = toDataUrl(product.image);
      var name = product.product_name || "Unknown Product";
      var category = product.category || "N/A";
      var article = product.article_type || product.sub_category || "N/A";
      var color = product.color || "N/A";

      var card = document.createElement("article");
      card.className = "product-card";
      card.innerHTML =
        '<div class="product-image-wrap">' +
        '<img src="' + escapeHtml(imageSrc) + '" alt="' + escapeHtml(name) + '" loading="lazy" />' +
        "</div>" +
        '<div class="product-body">' +
        '<span class="match-badge">' + escapeHtml(product.similarity_score) + '% Match</span>' +
        '<h3>' + escapeHtml(name) + "</h3>" +
        '<p class="product-meta">' + escapeHtml(category) + " • " + escapeHtml(article) + " • " + escapeHtml(color) + "</p>" +
        "</div>";

      singleResultsGrid.appendChild(card);
    });
  }

  function renderOutfitResults(data) {
    clearResults();
    if (outfitResults) outfitResults.classList.remove("is-hidden");

    var detected = Array.isArray(data.detected_items) ? data.detected_items : [];
    var analysisMs = Number(data.analysis_time_ms || 0);

    if (outfitSummary) {
      var boxed = toDataUrl(data.outfit_image_with_boxes || "");
      outfitSummary.innerHTML =
        '<h3 style="margin:0 0 8px;">Detected ' + detected.length + ' item(s)</h3>' +
        '<p style="margin:0 0 10px;color:#5b6475;">Analysis completed in ' + Math.round(analysisMs) + 'ms</p>' +
        (boxed
          ? '<div class="product-image-wrap" style="border-radius:10px;overflow:hidden;"><img src="' + escapeHtml(boxed) + '" alt="Outfit with detections" style="height:340px;" /></div>'
          : "");
    }

    if (detectedItems) {
      detectedItems.innerHTML = "";
      detected.forEach(function (item) {
        var crop = toDataUrl(item.cropped_image || "");
        var results = Array.isArray(item.search_results) ? item.search_results.slice(0, 6) : [];

        var gridHtml = results
          .map(function (r) {
            var rimg = toDataUrl(r.image || "");
            var meta = r.metadata || {};
            var title = meta.productDisplayName || "Unknown Product";
            var color = meta.baseColour || "N/A";
            return (
              '<article class="product-card">' +
              '<div class="product-image-wrap"><img src="' + escapeHtml(rimg) + '" alt="' + escapeHtml(title) + '" loading="lazy" /></div>' +
              '<div class="product-body">' +
              '<span class="match-badge">' + escapeHtml(r.similarity_score) + '% Match</span>' +
              '<h3>' + escapeHtml(title) + '</h3>' +
              '<p class="product-meta">' + escapeHtml(meta.articleType || "N/A") + " • " + escapeHtml(color) + "</p>" +
              "</div></article>"
            );
          })
          .join("");

        var node = document.createElement("div");
        node.className = "detected-item-card";
        node.innerHTML =
          '<div class="detected-item-top">' +
          '<img src="' + escapeHtml(crop) + '" alt="Detected item" />' +
          '<div><h3 style="margin:0 0 6px;">' + escapeHtml((item.category || "item").toUpperCase()) + '</h3>' +
          '<p style="margin:0;color:#5b6475;">Confidence: ' + Math.round(Number(item.confidence || 0) * 100) + '%</p></div>' +
          '</div>' +
          '<div class="detected-item-grid">' + gridHtml + '</div>';

        detectedItems.appendChild(node);
      });
    }

    if (outfitRecommendations) {
      var comp = data.outfit_recommendations && data.outfit_recommendations.complementary_items
        ? data.outfit_recommendations.complementary_items
        : [];

      if (!comp.length) {
        outfitRecommendations.innerHTML = '<p style="margin:0;">No additional outfit recommendations available.</p>';
      } else {
        var list = comp
          .map(function (entry) {
            var fromCat = (entry.detected_category || "item").toUpperCase();
            var block = (entry.recommendations || []).map(function (r) {
              var items = (r.items || []).slice(0, 3).map(function (it) {
                return '<li>' + escapeHtml(it.product_name || "Item") + ' (' + escapeHtml(it.color || "N/A") + ') - ' + escapeHtml(it.reason || "") + '</li>';
              }).join("");
              return '<p><strong>' + escapeHtml((r.category || "item").toUpperCase()) + ':</strong></p><ul>' + items + '</ul>';
            }).join("");
            return '<div style="margin-bottom:10px;"><h4 style="margin:0 0 6px;">Based on your ' + escapeHtml(fromCat) + ':</h4>' + block + '</div>';
          })
          .join("");

        outfitRecommendations.innerHTML = '<h3 style="margin:0 0 10px;">Complete the Look</h3>' + list;
      }
    }
  }

  function renderPreview(file) {
    var img = ensurePreviewImage();
    if (!img) return;
    var url = URL.createObjectURL(file);
    img.src = url;
    img.classList.remove("is-hidden");
    if (previewPlaceholder) previewPlaceholder.classList.add("is-hidden");
  }

  if (previewImage) {
    previewImage.addEventListener("error", function () {
      previewImage.src = "";
      previewImage.classList.add("is-hidden");
      if (previewPlaceholder) previewPlaceholder.classList.remove("is-hidden");
    });
  }

  updateMode("single");
  updateInputMode("upload");
})();
