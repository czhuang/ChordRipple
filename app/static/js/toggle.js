
$(document).on('click', '.toggle-button', function() {
    console.log('parent toggle, click!')
    $(this).toggleClass('toggle-button-selected');
});