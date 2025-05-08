---
title: How I created my first Chrome Extension via Iterative Prompting using Gemini 2.5 Pro
date: 2025-05-07 18:00:00 +0100
categories: [Gen AI, Gemini, Chrome Extension]
tags: [Generative AI, Chrome Extension, Gen AI, Gemini]
---

The process of learning a new language often involves repetitive tasks that, while necessary, can disrupt workflow. For Japanese learners utilizing WaniKani, looking up vocabulary encountered during general web browsing is one such task. I would love to have a chrome extension for this, however I have no previous experience building one. 

This post details the development of a Chrome extension designed to streamline the lookup process, achieved through an iterative dialogue with Gemini 2.5 Pro, demonstrating how targeted prompts can guide an AI to generate completely functional and refined code. The whole process took no more than 15 - 20 mins (Even this blog post is 99% Gemini generated).

## I. Project Objective and Initial AI Engagement

**Objective:** To create a Chrome extension that allows users to select Japanese text on a webpage and open a new tab directly to the corresponding WaniKani vocabulary page.

**Initial Prompt to AI:**
The engagement began with a clear, functional requirement:

> "Help me write and install a Chrome extension that does the following:
> 1. Allows the user to double click on a word.
> 2. Upon the double click, it copies the selected text, lets call it `selected_text`. It creates a url by adding `https://www.wanikani.com/vocabulary/` as a prefix to the `selected_text`.
> 3. It opens this url in a new tab."

**AI Model Response & Core Extension Structure:**
The AI responded by outlining the fundamental structure of a Chrome extension, identifying the need for two primary files:

1.  **`manifest.json`**: The extension's configuration file.
2.  **`content_script.js`**: The JavaScript file to be injected into web pages to handle the desired functionality.

The AI provided initial code for both:

**`manifest.json` (Initial Version):**
```json
{
  "manifest_version": 3,
  "name": "WaniKani Quick Lookup",
  "version": "1.0",
  "description": "Double-click a word to look it up on WaniKani.",
  "permissions": [
    "activeTab" // While window.open can work without it from user-initiated event, good for future.
  ],
  "content_scripts": [
    {
      "matches": ["<all_urls>"], // Injects into all URLs
      "js": ["content_script.js"]
    }
  ],
  "icons": { // Placeholder, led to initial error
    "128": "icon128.png"
  }
}
```
*Key takeaways from manifest:* `manifest_version: 3` is standard. `permissions` are declared. `content_scripts` defines where (`matches`) and what (`js`) scripts are injected.

**`content_script.js` (Initial Version):**
```javascript
document.addEventListener('dblclick', function() {
  const selectedText = window.getSelection().toString().trim();

  if (selectedText) {
    // The AI correctly identified the need for URL encoding for special characters.
    const encodedText = encodeURIComponent(selectedText);
    const wanikaniUrl = `https://www.wanikani.com/vocabulary/${encodedText}`;
    window.open(wanikaniUrl, '_blank'); // Opens in new tab
  }
});
```
*Key takeaways from content script:* Uses `dblclick` listener. `window.getSelection().toString().trim()` retrieves selected text. `encodeURIComponent()` is critical for handling non-ASCII characters in URLs. `window.open()` achieves the new tab functionality.

The AI also provided instructions for loading the extension in Chrome via `chrome://extensions` and enabling "Developer mode."

## II. Iteration 1: Addressing Errors and Refining Triggers

**Problem 1: Icon Error**
Upon attempting to load the unpacked extension, Chrome reported an error: `Could not load icon 'icon128.png'`.

**Prompt to AI:**
> "It throws an error regarding `icon128.png`. Can you create a sample `icon128.jpg` for me?"

**AI Model Response:**
The AI clarified that PNG is generally preferred for icons due to transparency support. It offered two solutions:
1.  Provide SVG code for a simple icon (e.g., text on a colored background) and instructions for online conversion to PNG.
2.  Instruct the user to remove the `icons` block from `manifest.json` if an icon was not immediately necessary.
This resolved the loading error.

**Problem 2: Trigger Refinement and Text Validation**
The initial double-click trigger was too broad and lacked specificity for Japanese text.

**Prompt to AI:**
> "Make the following changes:
> 1. Make sure that the text in `selectedText` is a Japanese text before moving forward.
> 2. Ctrl key + double click instead of just double-click should be the trigger."

**AI Model Response & Code Modification (`content_script.js` v2):**
The AI updated `content_script.js` to incorporate these requirements:

```javascript
document.addEventListener('dblclick', function(event) { // 'event' object now passed
  // 2. Check for Ctrl key (or Meta key for macOS compatibility)
  if (event.ctrlKey || event.metaKey) {
    const selectedText = window.getSelection().toString().trim();

    if (selectedText) {
      // 1. Check for Japanese characters using a Regular Expression
      // AI provided a comprehensive regex for Hiragana, Katakana, CJK Ideographs, and punctuation.
      const japaneseRegex = /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]/;

      if (japaneseRegex.test(selectedText)) {
        // Basic length check also suggested by AI (implicitly from previous version, retained)
        if (selectedText.length > 0 && selectedText.length < 50 && !selectedText.includes('\n')) {
          const encodedText = encodeURIComponent(selectedText);
          const wanikaniUrl = `https://www.wanikani.com/vocabulary/${encodedText}`;
          window.open(wanikaniUrl, '_blank');
        }
      }
    }
  }
});
```
*Key changes:*
*   The event listener now receives the `event` object.
*   `event.ctrlKey || event.metaKey` ensures the trigger is specific.
*   A `japaneseRegex` was introduced to validate the selected text, using Unicode character ranges. The `.test()` method efficiently checks for matches.

## III. Iteration 2: Expanding Functionality and Robustness

**Requirement: Adding Text Selection as a Trigger**
To enhance usability, a trigger based on simple text selection (on mouse release) was requested, in addition to Ctrl+DoubleClick.

**Prompt to AI:**
> "Take all these steps not only for Ctrl + double click but also for when a text is selected."

**AI Model Response & Code Modification (`content_script.js` v3 - Final Structure):**
The AI refactored the script significantly, introducing a shared processing function and a new event listener.

```javascript
let lastProcessedTimestamp = 0;
const debounceTime = 100; // milliseconds to prevent rapid successive calls

/**
 * Centralized function for processing selection and opening WaniKani tab.
 * @param {string} text The text to process.
 * @param {boolean} isExplicitTrigger For potential future differentiation or logging.
 * @returns {boolean} True if a tab was opened, false otherwise.
 */
function processAndOpenWaniKani(text, isExplicitTrigger = false) {
  const selectedText = text.trim();
  if (!selectedText) return false;

  const japaneseRegex = /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]/;
  if (!japaneseRegex.test(selectedText)) return false;

  if (selectedText.length === 0 || selectedText.length >= 50 || selectedText.includes('\n')) return false;

  // Debounce mechanism to avoid multiple triggers
  const now = Date.now();
  if (now - lastProcessedTimestamp < debounceTime) {
    return false;
  }

  const encodedText = encodeURIComponent(selectedText);
  const wanikaniUrl = `https://www.wanikani.com/vocabulary/${encodedText}`;
  window.open(wanikaniUrl, '_blank');
  lastProcessedTimestamp = now; // Update timestamp
  return true;
}

// Listener 1: Ctrl/Cmd + Double Click
document.addEventListener('dblclick', function(event) {
  if (event.ctrlKey || event.metaKey) {
    const currentSelection = window.getSelection().toString();
    if (currentSelection.trim()) {
        processAndOpenWaniKani(currentSelection, true);
    }
  }
});

// Listener 2: Text Selection (on mouse button release)
document.addEventListener('mouseup', function(event) {
  // Ensure it's the primary mouse button
  if (event.button !== 0) return;

  const currentSelectionText = window.getSelection().toString();

  if (currentSelectionText.trim()) {
    // Prevent triggering if selection is within an editable field
    const selection = window.getSelection();
    if (selection && selection.anchorNode) {
      let node = selection.anchorNode;
      let element = (node.nodeType === Node.TEXT_NODE) ? node.parentNode : node;
      let inEditableField = false;
      // Traverse up the DOM to check for input, textarea, or contentEditable
      while (element && element !== document.body && element.nodeType === Node.ELEMENT_NODE) {
        const tagName = element.tagName.toLowerCase();
        if (tagName === 'input' || tagName === 'textarea' || element.isContentEditable) {
          inEditableField = true;
          break;
        }
        element = element.parentNode;
      }
      if (inEditableField) return; // Skip if in editable field
    }
    processAndOpenWaniKani(currentSelectionText, false);
  }
});
```
*Key enhancements in v3:*
*   **`processAndOpenWaniKani` function:** Encapsulates the core logic (validation, URL construction, tab opening) for reusability (DRY principle).
*   **`mouseup` event listener:** Triggers on mouse button release after text selection.
*   **Editable Field Check:** The `mouseup` listener intelligently checks if the selection's anchor node is within an `<input>`, `<textarea>`, or an element with `contentEditable=true`. This prevents unintended lookups during text editing. The DOM traversal (`node.parentNode`) is a standard technique for this.
*   **Debounce Mechanism:** `lastProcessedTimestamp` and `debounceTime` (100ms) prevent near-simultaneous activations of both listeners (e.g., a Ctrl+DoubleClick also firing `mouseup`) from opening multiple tabs.

## IV. Final Extension Functionality and Installation

The resulting extension provides two methods for WaniKani lookup:
1.  **Ctrl/Cmd + Double-Click:** On any Japanese text.
2.  **Text Selection Release:** Select Japanese text; upon mouse button release (outside of editable fields), the lookup occurs.

{% include embed-video.html url='/assets/videos/wanikani-ext.mp4' %}

## V. Conclusion: AI as a Development Accelerator

This project demonstrates the utility of AI models like Gemini 2.5 Pro as interactive coding assistants. Through precise, iterative prompts, it was possible to:
*   Generate foundational code for a Chrome extension.
*   Debug issues based on error messages.
*   Implement specific functional requirements (trigger conditions, text validation).
*   Incorporate more nuanced logic (editable field checks, debouncing) that improve user experience.

The AI's ability to understand functional requirements, provide relevant code structures (like regex for Unicode ranges or DOM traversal patterns), and adapt to iterative feedback significantly accelerated the development process, especially for an individual who might not be deeply familiar with all aspects of Chrome extension development. The key was clear communication of requirements and incremental refinement based on the AI's output.
```