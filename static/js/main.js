$(document).ready(function () {
  $(document).keydown(function (event) {
    if (event.which == 82) {
      // 82 is the keycode for 'R'
      $.ajax({
        url: "/reset",
        type: "POST",
        success: function (response) {
          console.log("Game reset successfully");
        },
        error: function (error) {
          console.log("Error resetting game");
        },
      });
    }
  });
});
