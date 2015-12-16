

var moduleId = 'myModule';

angular.module(moduleId, ['ui.bootstrap']);
// testing rating


function makeRating(id, parent) {
    var controllerId = 'rating_' + id

    angular.module(moduleId).controller(controllerId, ['$scope', function ($scope) {
        $scope.rate = 2;
        $scope.max = 7;
        $scope.isReadonly = false;
        $scope.hoveringOver = function(value) {
            $scope.overStar = value;
            $scope.percent = value;
        };

        $scope.$watch('rate', function(value) {
            console.log('rating', value, controllerId)
            var ind = controllerId.split('_').pop()
            var text = $('#bookmarkTextSpan_'+ind).text()
            console.log('socket', socket)
            socket.emit('rating', value, ind, text)

        })

    }]);

    var ng_controller = $('<span>').attr('id', 'rating_'+id).attr('ng-controller', controllerId).addClass("ng-scope").appendTo(parent)
    ng_controller.attr('ng-hide', '1').css("margin-left", 6).css("margin-top", 5)
    var rate_model = $("<rating>").attr('ng-model', 'rate').attr({'max':"max", 'readonly':"isReadonly"})
    rate_model.attr({'on-hover':'hoveringOver(value)', 'on-leave':'overStar=null'}).appendTo(ng_controller)
    $('<span>').addClass('label').attr({'ng-class':"{'label-warning': percent<3, 'label-info': percent>=3 && percent<5, 'label-success': percent>=5}", "ng-show":"overStar && !isReadonly"}).text("{{percent}} stars").appendTo(ng_controller);
};


function makeLikeCross(id, parent) {
    var controllerId = 'likeCross' + id

    var clearButton = $('<button>').addClass('btn, btn-xs btn-default').text('clear').attr('ng-disabled', 'isReadonly')
    clearButton.css('line-height', 1.0)

    angular.module(moduleId).controller(controllerId, function ($scope) {
        $scope.like = 0;
        $scope.cross = 0;
        $scope.max = 1;
        $scope.isReadonly = false;

        $scope.$watch('like', function(value) {
            console.log('like', value)
        })

        $scope.$watch('cross', function(value) {
            console.log('cross', value)
        })

    });

    var ng_controller = $('<div>').addClass('ng-scope').attr('ng-controller', controllerId).appendTo(parent)

    // like rater
    var like_model = $("<rating ng-model='like'>").addClass('ng-scope').attr({'max':"max", 'readonly':"isReadonly"})
    like_model.attr({'state-on':"'glyphicon-heart'", 'state-off':"'glyphicon-heart-empty'"})
    like_model.attr({'on-hover':'hoveringOver(value)', 'on-leave':'overStar=null'}).appendTo(ng_controller)

     // cross rater
    var cross_model = $("<rating ng-model='cross'>").addClass('ng-scope').attr({'max':"max", 'readonly':"isReadonly"})
    cross_model.attr({'state-on':"'glyphicon-remove-sign'", 'state-off':"'glyphicon-remove-circle'"})
    cross_model.attr({'on-hover':'hoveringOver(value)', 'on-leave':'overStar=null'}).appendTo(ng_controller)

    clearButton.attr('ng-click', "cross=0; like=0").appendTo(ng_controller)

};

